# 参考校验：与官方 Qwen2 实现对照

状态：已完成（2026-11-16）。脚本：[`scripts/ref_check.py`](../scripts/ref_check.py)。

项目本身是纯 C++ 的；这个脚本只在开发期使用，用来回答一个问题：**“你怎么证明你的前向和分词是对的？”**

## 结论

| 校验项 | 结果 |
| --- | --- |
| chat 模板 + BPE 分词得到的 input ids | **35/35 完全一致** |
| 最后一个 prompt 位置的 logits | **最大绝对误差 5.82e-5**（logits 量级 21.3，相对误差约 2.7e-6） |
| greedy 生成前 16 个 token | **16/16 逐 token 一致** |

另一组提问（`1加1等于几？只回答数字。`，模型输出 `2` 后立即 EOS）同样通过：ids 39/39 一致，logits 最大误差 1.0e-4（量级 26.8），首个 token 一致。

## 校验的三件事

1. **分词与模板**：同一个问题，官方 `apply_chat_template` 与本项目 `QwenTokenizer` 生成的 token id 序列是否完全相同。
2. **前向数值**：给定同一组 id，官方 FP32 前向与项目 `Model::prefill` 在最后一个位置的 logits 差多少。
3. **生成行为**：argmax 贪心生成的前 N 个 token 是否逐 token 相同。

三项都通过，才能说“实现与官方一致”，而不只是“能跑出答案”。

## 做法

- **官方侧**：Windows conda 环境，`transformers 4.46.3` + `torch 2.3.1`（CPU），FP32。模型目录里没有 `config.json`，脚本按 Qwen2.5-0.5B 官方配置构造 `Qwen2Config`（896 / 4864 / 24 层 / 14 头 / 2 KV 头 / vocab 151936 / θ=1e6 / eps=1e-6 / tied embedding）。
- **项目侧**：WSL 里的 `bin/llm_rpc_generate --tokens <ids> 16`，同样以 FP32 跑。
- **桥接**：脚本在 Windows 上执行官方模型，通过 `wsl.exe` 调用 WSL 内的二进制；输入 ids 由官方 tokenizer 生成，双方吃**完全相同**的 id 序列，因此差异只可能来自前向实现本身。

## 口径细节

这两个细节不处理好，对比会“假失败”：

1. **EOS 不写入输出序列**。项目约定是命中 EOS 即结束、不把 EOS 放进 `tokens`；参考侧用同样口径，否则会比项目多出一个 token。
2. **不比较绝对数值，只比较同一输入下的相对误差**。logits 量级 20～27，1e-4 的绝对差对应约 3e-6 的相对差，属于不同实现（不同 matmul 顺序）在 FP32 下的正常范围。

## 能证明什么、不能证明什么

**能证明**：在 fp32、上下文 ≤ 256、单轮对话的条件下，embedding、RMSNorm、GQA、RoPE（θ=1e6、半区配对）、因果注意力、SwiGLU、tied 输出投影与官方实现数值一致；chat 模板与 BPE 合并规则一致。

**不能证明**：长上下文（> 256）的位置编码、多轮对话模板、BF16/量化路径、多线程下的稳定性。这些在 README 的“待完成”里仍然列着。

## 复现

```bash
# 1) 依赖（Windows conda 环境，torch 2.3 即可）
pip install --target D:\llmref transformers==4.46.3 tokenizers==0.20.3 "numpy<2" \
    "huggingface_hub<1.0" safetensors jinja2 regex tqdm pyyaml
set PYTHONPATH=D:\llmref

# 2) 运行（官方侧在 Windows，项目侧通过 wsl.exe 调 WSL 里的二进制）
python scripts/ref_check.py ^
    --model "\\wsl.localhost\Ubuntu2204\home\solar\ModelWeights\Qwen2.5-0.5B-Instruct" ^
    --distro Ubuntu2204 --root /home/solar/cpp_code/llm-rpc ^
    --question "请用一句话介绍杭州。" --max-new 16
```

纯 Linux 环境下省略 `--distro`，`--root` 指向仓库根目录即可。

> 版本约束：该 `tokenizer.json` 含 `ignore_merges` 字段，需要 `tokenizers >= 0.20`；因此不能用 transformers 4.44（它锁定 `tokenizers < 0.20`）。transformers 4.46 + tokenizers 0.20.3 与 torch 2.3 兼容。

## 局限

- 只在 CPU FP32 上对照，未覆盖 BF16 与量化路径。
- 上下文上限 256 是项目自身的限制，未验证更长位置编码。
- 只对照贪心解码；采样（top-k / temperature）未对照。
- 校验脚本依赖 Python 环境，不进入运行时依赖；C++ 二进制本身不需要任何 Python。
