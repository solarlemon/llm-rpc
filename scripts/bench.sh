#!/usr/bin/env bash
# 推理基准：单进程 / P-D 首请求（冷）/ P-D 预热（热）
#
# 输出：TTFT、decode tokens/s、快照字节，以及分段耗时
#       prefill 计算 / 快照编码 / 传输+序列化 / 快照导入 / 生成
#
# 用法：scripts/bench.sh [model.safetensors]
#
# 环境变量：
#   PROMPT_TOKENS=64   合成 prompt 的 token 数
#   PROMPT_IDS=""      直接指定逗号分隔的 token id（优先于 PROMPT_TOKENS）
#   GEN_TOKENS=64      生成 token 数
#   REPS=3             每个用例重复次数（取中位数）
#   WORK=/tmp/llm_bench
#   OUT=$WORK/result.md
#
# 说明：
#   * 默认用固定合成 token id（1000 起），保证三种模式输入完全一致；
#     合成 id 可能让模型提前输出 EOS，此时 tok/s 按实际 decode 步数计算；
#   * "冷" = worker 启动后的首批请求；"热" = 同一客户端进程连续请求（首次含 ZK 建连）；
#   * 跨请求 KV 复用（M3）尚未实现，热态不含 KV 命中，脚本末尾给出命中后的 TTFT 估算。
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$HOME/ModelWeights/Qwen2.5-0.5B-Instruct/model.safetensors}"
PROMPT_TOKENS="${PROMPT_TOKENS:-64}"
GEN_TOKENS="${GEN_TOKENS:-64}"
REPS="${REPS:-3}"
WORK="${WORK:-/tmp/llm_bench}"
OUT="${OUT:-$WORK/result.md}"

mkdir -p "$WORK/logs" "$(dirname "$OUT")"
JSONL="$WORK/runs.jsonl"
: > "$JSONL"

if [ -n "${PROMPT_IDS:-}" ]; then
    IDS="$PROMPT_IDS"
    PROMPT_TOKENS=$(awk -F, "{print NF}" <<< "$IDS")
else
    IDS=$(seq 1000 $((999 + PROMPT_TOKENS)) | awk 'NR>1{printf ","} {printf "%d", $0}')
fi
[ -n "$IDS" ] || { echo "无法生成 prompt token ids" >&2; exit 1; }

cat > "$WORK/record.py" <<'PY'
import json, os, sys
for i, line in enumerate(l.strip() for l in sys.stdin if l.strip()):
    data = json.loads(line)
    data["mode"] = os.environ["MODE"]
    data["case"] = os.environ["CASE"]
    data["rep"] = i
    with open(os.environ["JSONL"], "a") as f:
        f.write(json.dumps(data) + "\n")
PY

cat > "$WORK/report.py" <<'PY'
import json, os, statistics

rows = [json.loads(line) for line in open(os.environ["JSONL"])]

def pick(mode, case, key):
    vals = [r[key] for r in rows
            if r["mode"] == mode and r["case"] == case and key in r]
    return statistics.median(vals) if vals else 0.0

def fmt(v, nd=2):
    return f"{v:.{nd}f}"

out = []
out.append("| 模式 | TTFT | prefill 计算 | 快照编码 | 传输+序列化 | 快照导入 | decode/步 | decode tok/s | 快照 |")
out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

# ---- 单进程 ----
ttft_s = pick("single", "ttft", "prefill_ms")
pre_s = pick("single", "tput", "prefill_ms")
dec_s = pick("single", "tput", "decode_ms")
calls_s = pick("single", "tput", "decode_calls")
tps_s = calls_s / (dec_s / 1000) if dec_s > 0 and calls_s else 0.0
per_step_s = dec_s / calls_s if calls_s else 0.0
out.append(f"| 单进程 | {fmt(ttft_s)} ms | {fmt(pre_s)} ms | - | - | - | "
           f"{fmt(per_step_s)} ms | {fmt(tps_s)} | - |")

# ---- P/D ----
for mode, label in (("cold", "P/D 首请求（冷）"), ("hot", "P/D 预热（热）")):
    ttft = pick(mode, "ttft", "total_us") / 1000
    pre = pick(mode, "tput", "prefill_us") / 1000
    enc = pick(mode, "tput", "encode_us") / 1000
    rpc = pick(mode, "tput", "prefill_rpc_us") / 1000
    transfer = rpc - pre - enc
    imp = pick(mode, "tput", "import_us") / 1000
    gen = pick(mode, "tput", "generate_us") / 1000
    calls = pick(mode, "tput", "decode_forward_calls")
    snap = pick(mode, "tput", "snapshot_bytes") / 1024 / 1024
    tps = calls / (gen / 1000) if gen > 0 and calls else 0.0
    per_step = gen / calls if calls else 0.0
    out.append(f"| {label} | {fmt(ttft)} ms | {fmt(pre)} ms | {fmt(enc)} ms | {fmt(transfer)} ms | "
               f"{fmt(imp)} ms | {fmt(per_step)} ms | {fmt(tps)} | {fmt(snap)} MB |")

print("")
print("\n".join(out))
print("")

load_ms = pick("single", "ttft", "load_ms")
calls = pick("hot", "tput", "decode_forward_calls")
gen_tokens = int(os.environ["GEN_TOKENS"])
print(f"- prompt {os.environ['PROMPT_TOKENS']} tokens、生成 {gen_tokens} tokens，"
      f"每用例 {os.environ['REPS']} 次取中位数；单进程模型加载 {load_ms / 1000:.1f} s（不计入 TTFT）")
print("- TTFT：单进程为 prefill 计算时间；P/D 为客户端从启动到收到首 token 的端到端时间（两段 RPC 之和）")
print("- 传输+序列化 = Prefill RPC 往返 − 服务端 prefill 计算 − 快照编码")
print(f"- 热态实际 decode 步数中位数 {calls:.0f}（请求上限 {gen_tokens}，提前 EOS 会少于该值）")
hot_rows = sorted([r for r in rows if r["mode"] == "hot" and r["case"] == "tput"],
                  key=lambda r: r["rep"])
if hot_rows:
    first_rpc = hot_rows[0]["prefill_rpc_us"] / 1000
    steady = [r["prefill_rpc_us"] / 1000 for r in hot_rows if r["rep"] >= 1]
    steady_rpc = statistics.median(steady) if steady else first_rpc
    print(f"- 热态首请求 Prefill RPC {first_rpc:.1f} ms（含 ZK 会话建立），稳态 {steady_rpc:.1f} ms")

ttft = pick("hot", "ttft", "total_us") / 1000
pre = pick("hot", "ttft", "prefill_us") / 1000
enc = pick("hot", "ttft", "encode_us") / 1000
est = ttft - pre - enc
if est > 0:
    print(f"- 若实现跨请求 KV 复用（M3）：TTFT 预计降到约 {est:.0f} ms（跳过 prefill 计算与编码，估算值）")
PY

cleanup() {
    [ -f "$WORK/prefill.pid" ] && kill "$(cat "$WORK/prefill.pid")" 2>/dev/null
    [ -f "$WORK/decode.pid" ] && kill "$(cat "$WORK/decode.pid")" 2>/dev/null
}
trap cleanup EXIT

cd "$ROOT"
pkill -f llm_rpc_inference_worker 2>/dev/null
sleep 1

client() { ./bin/llm_rpc_inference_client -i bin/test.conf "$1" "$2" 60000 "$3" 2>/dev/null; }

run_single() { # $1=case $2=max_tokens $3=rep
    local out
    out=$(LLM_RPC_BENCH=1 ./bin/llm_rpc_generate "$MODEL" --tokens "$IDS" "$2" 2>/dev/null) \
        || { echo "单进程执行失败" >&2; return 1; }
    printf '%s' "$out" | MODE=single CASE="$1" REP="$3" JSONL="$JSONL" python3 "$WORK/record.py"
}

run_rpc() { # $1=mode $2=case $3=max_tokens $4=repeats
    local out
    out=$(LLM_RPC_REPEAT="$4" ./bin/llm_rpc_inference_client -i bin/test.conf "$IDS" "$3" 60000 "$((1000 + $4))" 2>/dev/null) \
        || { echo "客户端执行失败" >&2; return 1; }
    [ -n "$out" ] || { echo "客户端无输出" >&2; return 1; }
    printf '%s' "$out" | MODE="$1" CASE="$2" REP="$4" JSONL="$JSONL" python3 "$WORK/record.py"
}

echo "[1/4] 单进程基线（每次重新加载模型，共 $REPS 轮）..."
for i in $(seq 1 "$REPS"); do
    run_single ttft 1 "$i" || exit 1
    run_single tput "$GEN_TOKENS" "$i" || exit 1
    echo "      第 $i 轮完成"
done

echo "[2/4] 启动 worker ..."
./bin/llm_rpc_inference_worker prefill "$MODEL" -i bin/prefill.conf > "$WORK/logs/prefill.log" 2>&1 &
echo $! > "$WORK/prefill.pid"
./bin/llm_rpc_inference_worker decode "$MODEL" -i bin/decode.conf > "$WORK/logs/decode.log" 2>&1 &
echo $! > "$WORK/decode.pid"

ready=0
for i in $(seq 1 90); do
    if grep -q 'RpcProvider start service' "$WORK/logs/prefill.log" 2>/dev/null &&
       grep -q 'RpcProvider start service' "$WORK/logs/decode.log" 2>/dev/null; then
        echo "      worker ready (${i}s)"; ready=1; break
    fi
    sleep 1
done
[ "$ready" -eq 1 ] || { echo "worker 未就绪"; tail -20 "$WORK/logs/prefill.log"; exit 1; }

echo "[3/4] 冷态：worker 就绪后的首批请求"
run_rpc cold ttft 1 1 || exit 1
run_rpc cold tput "$GEN_TOKENS" 1 || exit 1

echo "[4/4] 热态：同一客户端进程连续请求 $((REPS + 1)) 次（首次含 ZK 建连，取中位数）"
run_rpc hot ttft 1 "$((REPS + 1))" || exit 1
run_rpc hot tput "$GEN_TOKENS" "$((REPS + 1))" || exit 1
echo "      完成"

echo
PROMPT_TOKENS="$PROMPT_TOKENS" GEN_TOKENS="$GEN_TOKENS" REPS="$REPS" JSONL="$JSONL" \
    python3 "$WORK/report.py" | tee "$OUT"
echo
echo "结果: $OUT"
echo "原始: $JSONL"
