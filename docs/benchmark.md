# 推理基准：单进程与 Prefill/Decode 分离

状态：三模式基准已完成（2026-11-16）。脚本：[`scripts/bench.sh`](../scripts/bench.sh)。分段计时的实现见 [`src/inference/inference_worker.cpp`](../src/inference/inference_worker.cpp)、[`src/inference/inference_coordinator.cpp`](../src/inference/inference_coordinator.cpp)、[`src/rpc_channel.cc`](../src/rpc_channel.cc)。

## 目标

回答三个问题：

1. Prefill/Decode 拆成两个进程后，KV 交接到底花多少钱？
2. 这个开销里哪一段是主要矛盾，能不能消掉？
3. 单机上的分离式推理，比单进程慢多少？

## 环境与输入

| 项 | 值 |
| --- | --- |
| 环境 | WSL2 Ubuntu 22.04，单机 |
| 模型 | Qwen2.5-0.5B-Instruct，FP32，256 上下文 |
| 并发 | 单推理槽位（一个 worker 同时只跑一个请求） |
| prompt | 固定合成 token id `1000..1063`（64 个），保证三种模式输入完全一致 |
| 生成 | 上限 64，实测 63 步（贪心，未触发 EOS） |
| 统计 | 每用例 3 次取中位数；热态用同一客户端进程连发 4 次 |

## 方法

### 三种模式

| 模式 | 做法 | 说明 |
| --- | --- | --- |
| 单进程 | `llm_rpc_generate --tokens` | 一个进程内 prefill + decode，作为基线 |
| P/D 冷 | worker 就绪后的首批请求 | 客户端冷启动：建 ZK 会话、查服务、建连 |
| P/D 热 | 同一客户端进程连续请求 | 复用发现结果与连接池，取中位数 |

### 指标定义

- **TTFT**：从发起请求到拿到首个 token。单进程模式为 prefill 计算时间；P/D 模式为两段 RPC 之和（不含进程启动）。
- **decode/步、tok/s**：服务端 `generate_us / decode_forward_calls`，避免把首 token 和 EOS 提前结束算进去。
- **分段耗时**：服务端回传 `prefill_us`（前向计算）、`encode_us`（快照编码）、`import_us`（快照导入）、`generate_us`（生成）；客户端侧统计 `prefill_rpc_us`、`decode_rpc_us`。
- **传输+序列化** = `prefill_rpc_us − prefill_us − encode_us`，包含线程池派发、protobuf 序列化与拷贝、TCP 收发。

### 开关

| 变量 | 作用 |
| --- | --- |
| `LLM_RPC_REPEAT=N` | 同一客户端进程连续发起 N 次请求（长驻客户端模式） |
| `LLM_RPC_TRACE=1` | 把每次 RPC 拆成 discovery / borrow / send / recv+parse 打到 stderr |
| `LLM_RPC_BENCH=1` | 单进程模式跳过 151936 维 logits 打印 |

## 结果

TTFT 与分段耗时来自两组不同 `max_new` 的运行（前者 `max_new=1` 只测首 token，后者 `max_new=64`），因此分开放，避免跨表相减。

### 端到端

| 模式 | TTFT | decode/步 | decode tok/s |
| --- | ---: | ---: | ---: |
| 单进程 | 14316 ms | 225.9 ms | 4.43 |
| P/D 冷 | 14403 ms | 228.1 ms | 4.38 |
| P/D 热 | 14323 ms | 225.5 ms | 4.43 |

> 单进程的 TTFT 即 prefill 计算；P/D 的 TTFT 为分词 + 两段 RPC 之和。

### 分段耗时（`max_new=64`）

| 模式 | prefill 计算 | 快照编码 | 传输+序列化 | 快照导入 | 快照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单进程 | 14351 ms | - | - | - | - |
| P/D 冷 | 14351 ms | 6.9 ms | 63.2 ms | 9.5 ms | 1.50 MB |
| P/D 热 | 14336 ms | 7.9 ms | **25.0 ms** | 7.4 ms | 1.50 MB |

**正确性**：三种模式对同一 prompt 的输出 token 序列逐 token 完全相同（63 步，`finish_reason=length`）。

**快照大小**：1.50 MB ÷ 64 token = 24.0 KB/token，与理论值 `24 层 × 2 个 KV 头 × 64 维 × 4 B × 2(K/V)` 一致。

## 分段归因

### 冷态与热态差在哪

`LLM_RPC_TRACE=1` 下单次请求：

```
[channel] PrefillService discovery=38299us borrow=524us send=31us recv+parse=4100960us bytes=393294
[channel] DecodeService  discovery=1951us  borrow=631us send=732us recv+parse=1927337us bytes=36
```

- 第一次调用的 `discovery` 包含 **ZooKeeper 会话建立**，这一轮 38 ms；另一轮测得 75 ms（见下方长驻客户端数据），说明它与调度相关、在几十毫秒量级波动。第二次调用只查服务，约 2 ms。
- 长驻客户端（`LLM_RPC_REPEAT=3`）的逐次数据：

```
iter0  discovery=75320us  borrow=568us
iter1  discovery=2us      borrow=1us
iter2  discovery=2us      borrow=0us
```

结论：**ZK 建连是「每进程一次」而不是「每请求一次」**，只有长驻进程能把它摊掉；进程内已缓存一致性哈希环，后续请求连 ZK 都不碰。

### 热态 25 ms 的构成

先明确口径：25 ms 是 **Prefill 回包路径**的开销，等于 `prefill_rpc − prefill − encode`，**不含**快照编码（7.9 ms，已单独计时）与 Decode 侧的导入（7.4 ms）。完整交接约为 7.9 + 25.0 + 7.4 ≈ **40 ms**（占端到端 0.28%）。

热态逐次请求的该开销：

| 轮次 | 额外开销 |
| --- | ---: |
| 0（含 ZK 建连） | 124.0 ms |
| 1 | 21.8 ms |
| 2 | 28.3 ms |
| 3 | 21.6 ms |

它由两块构成：

1. **1.5 MB 快照的链路搬运**。同机 TCP loopback 实测 66～290 MB/s（波动很大），1.5 MB 对应 5～23 ms。
2. **回包路径上的 3～4 次 1.5 MB 拷贝**：回复序列化 → 拼 4 字节长度头 → Muduo 发送缓冲 → 客户端 `ParseFromArray`。

## 结论

1. **交接开销可忽略。** 热态回包路径 25 ms、完整交接约 40 ms（编码 7.9 + 回包 25.0 + 导入 7.4），占端到端 14.3 s 的 0.17%～0.28%；单机 P/D 与单进程的端到端差异在噪声内。分离的价值不在单机加速，而在按阶段独立扩缩容与故障隔离。
2. **服务发现是「每进程」成本，不是「每请求」成本。** 这是三态对比里唯一被消掉的量级（38 ms → 2 µs）。
3. **decode 受内存带宽约束。** 每步要读完 1.96 GB FP32 权重，225.5 ms/步 → 有效带宽约 **8.7 GB/s**，与带宽受限模型吻合。因此单请求提速要靠量化，吞吐要靠批处理，而不是多线程。
4. **prefill 是当前唯一的主要成本。** 64 token 需要 14.3 s，约 224 ms/token，这是 naive FP32 逐 token 前向的直接结果；也是 SIMD / 批处理 / chunked prefill 的用武之地。

## 负结果：为什么没有改用 Unix domain socket

同机 TCP loopback 的吞吐上限看起来是瓶颈，因此评估过 AF_UNIX：

| 链路 | 吞吐 | 1.5 MB 单次 |
| --- | ---: | ---: |
| TCP 127.0.0.1 | 66–290 MB/s（多次波动） | 5–23 ms |
| AF_UNIX | 643 MB/s（单次采样） | 2.3 ms |

不做的三个理由：

1. **收益被端到端淹没**：回包路径从 25 ms 降到约 10 ms（传输 17→2.3 ms），占端到端由 0.17% 变 0.07%，用户视角的 TTFT 不变。
2. **成本不低**：Muduo 只提供 `InetAddress` / `TcpServer` / `TcpClient`，不支持 AF_UNIX。要做就得另写一套基于 epoll 的监听与连接管理，并把帧解析、派发逻辑复制一份，形成两套传输路径。
3. **测量噪声与收益同量级**：同一组 buffer 配置复测两次，吞吐分别是 93 MB/s 和 290 MB/s；在这种环境下宣称"快 7 倍"不可靠。

**决定**：暂缓，等 M3（跨请求 KV 复用）落地。届时 prefill 可被跳过，TTFT 预计从 14.3 s 降到约 60 ms，交接将占到 TTFT 的约 40%，那时换 UDS 的收益才可测、可讲。

## 局限

- 单机 WSL2，虚拟网络与调度噪声大，绝对值不具备横向可比性，只用于同一环境下的相对比较。
- prompt 是合成 token，不是自然语言；只影响语义，不影响计算量。
- 贪心采样、无批处理、无前缀复用。
- 三种模式的输出一致性在本文验证；与官方 transformers 的数值对照见 [参考校验](reference-check.md)。

## 复现

```bash
cmake --build build -j4
./scripts/bench.sh                                  # 默认 64/64，3 轮
PROMPT_TOKENS=16 GEN_TOKENS=8 REPS=2 ./scripts/bench.sh
LLM_RPC_REPEAT=3 LLM_RPC_TRACE=1 ./bin/llm_rpc_inference_client \
    -i bin/test.conf "1000,1001,1002" 8 60000 1
```

结果写入 `/tmp/llm_bench/result.md`，原始数据在 `/tmp/llm_bench/runs.jsonl`。
