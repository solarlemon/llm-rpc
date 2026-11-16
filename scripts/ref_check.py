#!/usr/bin/env python3
"""参考校验：官方 HuggingFace 实现 vs 本项目的 C++ 实现。

项目本身是纯 C++ 的，本脚本只在开发期使用，用于回答
"你怎么证明前向和分词是对的"。

校验三件事：
  1. chat 模板 + BPE 分词得到的 input ids 是否与官方一致
  2. 最后一个 prompt 位置的 logits 与官方差多少
  3. greedy 生成的前 N 个 token 是否与官方逐 token 一致

Windows + WSL 用法（模型和二进制都在 WSL 里）：
  python scripts/ref_check.py \
      --model '\\\\wsl.localhost\\Ubuntu2204\\home\\solar\\ModelWeights\\Qwen2.5-0.5B-Instruct' \
      --distro Ubuntu2204 --root /home/solar/cpp_code/llm-rpc

纯 Linux 用法：
  python3 scripts/ref_check.py --model ~/ModelWeights/Qwen2.5-0.5B-Instruct --root .
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
EOS_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>

# 权重目录里只有 model.safetensors / tokenizer*，没有 config.json，
# 因此按 Qwen2.5-0.5B 官方配置构造；换模型时用 --config 覆盖。
DEFAULT_CONFIG = dict(
    vocab_size=151936, hidden_size=896, intermediate_size=4864,
    num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
    rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True,
    hidden_act="silu", max_position_embeddings=32768,
)


def split_wsl_path(path):
    """把 \\\\wsl.localhost\\<distro>\\a\\b 映射成 (<distro>, '/a/b')；非 WSL 路径返回 (None, path)。"""
    raw = path.replace("/", "\\")
    for prefix in ("\\\\wsl.localhost\\", "\\\\wsl$\\"):
        if raw.startswith(prefix):
            rest = raw[len(prefix):]
            distro, _, tail = rest.partition("\\")
            return distro, "/" + tail.replace("\\", "/")
    return None, path


class CppRunner:
    def __init__(self, root, distro, dump_path):
        self.root = root
        self.distro = distro
        self.dump_path = dump_path

    def _run(self, argv, **kwargs):
        if self.distro:
            cmd = "cd %s && %s" % (shlex.quote(self.root), " ".join(shlex.quote(a) for a in argv))
            return subprocess.run(["wsl.exe", "-d", self.distro, "bash", "-lc", cmd], **kwargs)
        return subprocess.run([os.path.join(self.root, argv[0]), *argv[1:]],
                              cwd=self.root, **kwargs)

    def prompt_ids(self, model_dir, question):
        env = {"LLM_RPC_DUMP_PROMPT": self.dump_path}
        if self.distro:
            argv = ["env", "LLM_RPC_DUMP_PROMPT=" + self.dump_path,
                    "./bin/llm_rpc_inference_client",
                    "-i", "bin/test.conf", "--chat", model_dir,
                    question, "4", "60000", "1"]
            self._run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            windows_path = "\\\\wsl.localhost\\%s%s" % (self.distro, self.dump_path.replace("/", "\\"))
        else:
            argv = ["./bin/llm_rpc_inference_client", "-i", "bin/test.conf", "--chat",
                    model_dir, question, "4", "60000", "1"]
            self._run(argv, env=dict(os.environ, **env),
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            windows_path = self.dump_path
        if not os.path.exists(windows_path):
            raise SystemExit("客户端没有导出 prompt ids：%s" % windows_path)
        with open(windows_path) as handle:
            return [int(x) for x in handle.read().strip().split(",")]

    def generate(self, safetensors, ids, max_new):
        argv = ["./bin/llm_rpc_generate", safetensors, "--tokens",
                ",".join(str(i) for i in ids), str(max_new)]
        proc = self._run(argv, capture_output=True, text=True, check=True)
        return json.loads(proc.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录（含 model.safetensors）")
    ap.add_argument("--root", default=".", help="仓库根目录（WSL 内路径或本地路径）")
    ap.add_argument("--distro", default="", help="WSL 发行版名；为空表示在当前系统直接执行")
    ap.add_argument("--question", default="1加1等于几？只回答数字。")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--config", default="", help="可选：官方 config.json 路径")
    args = ap.parse_args()

    distro, linux_model = split_wsl_path(args.model)
    distro = args.distro or distro
    windows_model = args.model if not distro else args.model
    if distro:
        linux_model = linux_model.rstrip("/")

    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    # ---- 官方配置与分词 ----
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            cfg_dict = json.load(handle)
    else:
        cfg_dict = dict(DEFAULT_CONFIG)
    cfg = Qwen2Config(**cfg_dict)

    tokenizer = AutoTokenizer.from_pretrained(windows_model)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": args.question}]
    ids_ref = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True)

    runner = CppRunner(args.root, distro, "/tmp/ref_check_prompt.txt")
    ids_cpp = runner.prompt_ids(linux_model, args.question)

    # ---- 官方前向 ----
    model = Qwen2ForCausalLM.from_pretrained(windows_model, config=cfg,
                                             torch_dtype=torch.float32)
    model.eval()

    eos_hit = False
    with torch.no_grad():
        out = model(torch.tensor([ids_ref]))
        logits_ref = out.logits[0, -1].float().numpy()
        greedy_ref, past = [], out.past_key_values
        last = out.logits[0, -1]
        for _ in range(args.max_new):
            nxt = int(torch.argmax(last))
            # 与项目约定一致：命中 EOS 只表示结束，不写入输出序列
            if nxt in EOS_IDS:
                eos_hit = True
                break
            greedy_ref.append(nxt)
            cur = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past, last = cur.past_key_values, cur.logits[0, -1]

    # ---- 本项目 C++ 实现 ----
    cpp = runner.generate(linux_model + "/model.safetensors", ids_ref, args.max_new)
    logits_cpp = cpp["prefill_logits"]
    greedy_cpp = cpp["output_ids"][:args.max_new]

    import numpy as np
    ref = np.asarray(logits_ref, dtype=np.float32)
    ours = np.asarray(logits_cpp, dtype=np.float32)
    diff = np.abs(ref - ours)
    top1_ref = int(np.argmax(ref))
    top1_cpp = int(np.argmax(ours))
    prefix = 0
    for a, b in zip(greedy_ref, greedy_cpp):
        if a != b:
            break
        prefix += 1

    report = {
        "config": {k: cfg_dict[k] for k in ("hidden_size", "num_hidden_layers",
                                            "num_attention_heads", "num_key_value_heads",
                                            "vocab_size", "rope_theta", "rms_norm_eps")},
        "tokenizer": {
            "official_len": len(ids_ref), "cpp_len": len(ids_cpp),
            "match": list(ids_ref) == list(ids_cpp),
            "first_diff": next((i for i, (a, b) in enumerate(zip(ids_ref, ids_cpp)) if a != b), -1),
            "ids": list(ids_ref),
        },
        "logits": {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "ref_scale": float(np.abs(ref).max()),
            "top1_ref": top1_ref, "top1_cpp": top1_cpp,
            "top1_match": top1_ref == top1_cpp,
            "ref_logit_at_top1": float(ref[top1_ref]),
        },
        "greedy": {
            "ref": [int(x) for x in greedy_ref],
            "cpp": [int(x) for x in greedy_cpp],
            "prefix_match": prefix,
            "length": len(greedy_ref),
            "eos_hit_ref": eos_hit,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n摘要: tokenizer 一致=%s (%d ids) | logits max_abs=%.2e (量级 %.1f) | "
          "greedy 一致=%d/%d | 参考 EOS=%s"
          % (report["tokenizer"]["match"], len(ids_ref),
             report["logits"]["max_abs"], report["logits"]["ref_scale"],
             prefix, len(greedy_ref), eos_hit))

    ok = (report["tokenizer"]["match"] and report["logits"]["top1_match"]
          and prefix == len(greedy_ref))
    print("\n结果:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
