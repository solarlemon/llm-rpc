#!/usr/bin/env bash
# 验证「客户端断开 -> 服务端释放推理槽位」。
# 前置：ZooKeeper 已启动，bin/prefill.conf 与 bin/decode.conf 已配好不同端口。
# 用法：scripts/cancel_demo.sh [model.safetensors 路径]
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$HOME/ModelWeights/Qwen2.5-0.5B-Instruct/model.safetensors}"
TOKDIR="$(dirname "$MODEL")"
WORK=/tmp/llm_cancel_demo
LOG="$WORK/logs"
mkdir -p "$LOG"

cleanup() {
    [ -f "$WORK/prefill.pid" ] && kill "$(cat "$WORK/prefill.pid")" 2>/dev/null
    [ -f "$WORK/decode.pid" ] && kill "$(cat "$WORK/decode.pid")" 2>/dev/null
}
trap cleanup EXIT

cd "$ROOT"
pkill -f llm_rpc_inference_worker 2>/dev/null
sleep 1

LONG='请写一篇八百字的散文，题目是我的家乡。要求语言优美，结构清晰，情感真挚。'
SHORT='1加1等于几？只回答数字。'

CLIENT=(./bin/llm_rpc_inference_client -i bin/test.conf --chat "$TOKDIR")

echo "[1/5] 启动 worker ..."
./bin/llm_rpc_inference_worker prefill "$MODEL" -i bin/prefill.conf > "$LOG/prefill.log" 2>&1 &
echo $! > "$WORK/prefill.pid"
./bin/llm_rpc_inference_worker decode "$MODEL" -i bin/decode.conf > "$LOG/decode.log" 2>&1 &
echo $! > "$WORK/decode.pid"

ready=0
for i in $(seq 1 90); do
    if grep -q 'RpcProvider start service' "$LOG/prefill.log" 2>/dev/null &&
       grep -q 'RpcProvider start service' "$LOG/decode.log" 2>/dev/null; then
        echo "      worker ready (${i}s)"; ready=1; break
    fi
    sleep 1
done
[ "$ready" -eq 1 ] || { echo "      worker 未就绪"; tail -20 "$LOG/prefill.log"; exit 1; }

echo
echo "[2/5] 场景 A：槽位被占用（不取消），新请求应立即被拒绝"
"${CLIENT[@]}" "$LONG" 4 60000 7 > "$WORK/a1.out" 2>&1 &
A_PID=$!
sleep 6
start=$(date +%s%N)
timeout 15 "${CLIENT[@]}" "$SHORT" 4 60000 100 > "$WORK/a2.out" 2> "$WORK/a2.err"
end=$(date +%s%N)
if grep -q 'inference worker busy' "$WORK/a2.err"; then
    echo "      结果: busy（符合预期），往返 $(( (end - start) / 1000000 )) ms（含客户端启动）"
else
    echo "      结果: 未返回 busy"; tail -2 "$WORK/a2.err"
fi
kill -9 $A_PID 2>/dev/null; wait $A_PID 2>/dev/null
sleep 1

echo
echo "[3/5] 场景 B：prefill 中途断连，槽位应迅速释放"
"${CLIENT[@]}" "$LONG" 4 60000 9 > "$WORK/b1.out" 2>&1 &
B_PID=$!
sleep 6
kill -9 $B_PID 2>/dev/null; wait $B_PID 2>/dev/null
sleep 2
start=$(date +%s%N)
timeout 120 "${CLIENT[@]}" "$SHORT" 4 60000 101 > "$WORK/b2.out" 2> "$WORK/b2.err"
end=$(date +%s%N)
if grep -q 'inference worker busy' "$WORK/b2.err"; then
    echo "      结果: busy（槽位未释放，异常）"
else
    echo "      结果: 成功 -> $(tr -d '\n' < "$WORK/b2.out")，耗时 $(( (end - start) / 1000000 )) ms"
fi

echo
echo "[4/5] 场景 C：decode 中途断连（长生成被中断）"
"${CLIENT[@]}" "$LONG" 180 60000 11 > "$WORK/c1.out" 2>&1 &
C_PID=$!
sleep 22                                   # 等 prefill 完成、进入 decode
kill -9 $C_PID 2>/dev/null; wait $C_PID 2>/dev/null
sleep 2
start=$(date +%s%N)
timeout 120 "${CLIENT[@]}" "$SHORT" 4 60000 102 > "$WORK/c2.out" 2> "$WORK/c2.err"
end=$(date +%s%N)
if grep -q 'inference worker busy' "$WORK/c2.err"; then
    echo "      结果: busy（槽位未释放，异常）"
else
    echo "      结果: 成功 -> $(tr -d '\n' < "$WORK/c2.out")，耗时 $(( (end - start) / 1000000 )) ms"
fi

echo
echo "[5/5] 服务端取消日志（detected = 断连到被推理线程观察到的延迟）"
echo "  prefill:"; grep 'cancelled:' "$LOG/prefill.log" | sed 's/^/    /' || echo "    (无)"
echo "  decode :"; grep 'cancelled:' "$LOG/decode.log"  | sed 's/^/    /' || echo "    (无)"
