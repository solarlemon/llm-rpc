#include "inference_rpc.pb.h"
#include "snapshot.hpp"
#include "generation.hpp"
#include "rpc_application.h"
#include "rpc_provider.h"
#include "rpc_controller.h"
#include "rpc_transport.h"

#include <chrono>
#include <iostream>
#include <mutex>

namespace
{
    using namespace llm_rpc;
    using namespace llm_rpc::inference;

    using Clock = std::chrono::steady_clock; // 分段计时统一用单调时钟

    // 把时钟间隔折算成微秒，回传给客户端做分段统计
    std::uint32_t to_us(Clock::duration span)
    {
        return static_cast<std::uint32_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(span).count());
    }

    // 通用工作流包装器：处理互斥、取消（断连/超时）、异常捕获、回调
    template <class Reply, class Work>
    void RunWorker(std::mutex &slot, google::protobuf::RpcController *controller,
                   std::uint32_t timeout, Reply *reply,
                   google::protobuf::Closure *done, Work work)
    {
        const auto started = std::chrono::steady_clock::now();
        {
            // 1. 尝试加锁（std::try_to_lock），不阻塞等待。
            // 因为大模型占显存大，一个 Worker 进程通常只允许同时跑一个请求。
            std::unique_lock<std::mutex> lock(slot, std::try_to_lock);
            try
            {
                if (!lock)
                    throw std::runtime_error("inference worker busy");

                // 2. 校验超时时间（不能为 0，也不能超过 60 秒）
                if (!timeout || timeout > 60000)
                    throw std::invalid_argument("invalid inference timeout");

                // 3. 取消条件有两个来源：
                //    a) 客户端断开连接（IO 线程置位 CancelToken，毫秒级生效）
                //    b) 超时 deadline（兜底：网络分区等场景不会收到 FIN）
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::milliseconds(timeout);
                const Cancel cancelled = [controller, deadline]
                {
                    return (controller && controller->IsCanceled()) ||
                           std::chrono::steady_clock::now() >= deadline;
                };

                // 4. 执行真正的推理工作；内部每层/每 token 会检查取消
                work(cancelled);

                // 5. 结束后若已取消，结果作废（客户端已经不要了），但槽位照常释放
                if (cancelled())
                    throw std::runtime_error("inference cancelled");
            }
            catch (const std::exception &error)
            {
                // 6. 捕获任何异常，清空 Reply，并把错误信息返回给客户端
                const bool disconnected = controller && controller->IsCanceled();
                reply->Clear();
                reply->set_error(disconnected
                                     ? "request cancelled: client disconnected"
                                     : error.what());
                if (disconnected)
                {
                    const auto now = std::chrono::steady_clock::now();
                    const auto *owner = dynamic_cast<const RpcController *>(controller);
                    const auto stamp = owner ? owner->GetCancelToken() : nullptr;
                    const auto detected =
                        (stamp && stamp->DisconnectedAt())
                            ? std::chrono::duration_cast<std::chrono::milliseconds>(
                                  now.time_since_epoch() -
                                  std::chrono::steady_clock::duration(stamp->DisconnectedAt()))
                                  .count()
                            : -1LL;
                    std::clog << "[worker] cancelled: detected " << detected
                              << " ms after disconnect, slot held "
                              << std::chrono::duration_cast<std::chrono::milliseconds>(
                                     now - started)
                                     .count()
                              << " ms\n";
                }
            }
        } // 释放锁，允许下一个请求进入

        // 7. 执行 RPC 回调，把 Reply 发送回客户端（客户端已断开时由 provider 跳过发送）
        done->Run();
    }

    // Prefill 工作节点：计算 KV Cache 并打包
    class PrefillWorker final : public PrefillService
    {
    public:
        explicit PrefillWorker(const std::string &path) : model_(path) {}

        void RunPrefill(google::protobuf::RpcController *controller,
                        const PrefillRequest *request,
                        PrefillReply *reply, google::protobuf::Closure *done) override
        {
            RunWorker(slot_, controller, request->timeout_ms(), reply, done,
                      [&](const Cancel &cancelled)
                      {
                          // 1. 校验 Prompt 长度
                          if (request->prompt_size() <= 0 ||
                              request->prompt_size() > model_.config().capacity)
                              throw std::invalid_argument("invalid prompt length");

                          // 2. 把 Protobuf 的数组转成 C++ 的 vector
                          const std::vector<TokenId> prompt(request->prompt().begin(),
                                                            request->prompt().end());

                          // 3. 执行 Prefill 前向传播
                          // chunk_size 传 8：分块预填充，逐 token 写入 KV Cache
                          const auto prefill_begin = Clock::now();
                          KVCache cache(model_.cache_shape());
                          const auto logits = model_.prefill(prompt, cache, 8, cancelled);

                          // 4. 贪心采样出首 Token：
                          // logits 是长度为 vocab 的分布，取概率最高者作为首个生成 token
                          const auto first_token = Model::greedy(logits);
                          const auto prefill_end = Clock::now();

                          // 5. 序列化快照（限制最大包长 rpc_max_frame - 4096）
                          // 6MB 的编码是纯 CPU 工作，客户端断开后没必要白做
                          if (cancelled())
                              throw std::runtime_error("mini_cancelled");
                          reply->set_snapshot(KVSnapshot::encode(
                              model_, cache, first_token, request->request_id(),
                              rpc_max_frame - 4096, cancelled));
                          const auto encode_end = Clock::now();

                          // 6. 回传分段耗时：prefill 计算 与 快照编码
                          reply->set_prefill_us(to_us(prefill_end - prefill_begin));
                          reply->set_encode_us(to_us(encode_end - prefill_end));
                          reply->set_request_id(request->request_id());
                      });
        }

    private:
        Model model_;
        std::mutex slot_; // 保证同一时刻只有一个请求能占用模型
    };

    // Decode 工作节点：接收快照并连续生成
    class DecodeWorker final : public DecodeService
    {
    public:
        explicit DecodeWorker(const std::string &path) : model_(path) {}

        void RunDecode(google::protobuf::RpcController *controller,
                       const DecodeRequest *request,
                       DecodeReply *reply, google::protobuf::Closure *done) override
        {
            RunWorker(slot_, controller, request->timeout_ms(), reply, done,
                      [&](const Cancel &cancelled)
                      {
                          // 1. 从 RPC 包里解码出 KV Cache 快照
                          const auto import_begin = Clock::now();
                          auto state = KVSnapshot::decode(request->snapshot(), model_);
                          const auto import_end = Clock::now();

                          // 2. 校验 Request ID（防止路由错乱）
                          if (state.request_id != request->request_id())
                              throw std::invalid_argument("snapshot request ID mismatch");

                          // 3. 校验生成长度预算
                          if (request->max_tokens() == 0 ||
                              request->max_tokens() > model_.cache_shape().capacity + 1)
                              throw std::invalid_argument("invalid generation limit");
                          validate_budget(state.cache->valid_len(),
                                          model_.cache_shape().capacity,
                                          request->max_tokens());

                          // 4. 开始连续生成（复用 generation.hpp 的 generate 逻辑）
                          const auto generate_begin = Clock::now();
                          const auto result = generate(model_, *state.cache, state.first_token,
                                                       request->max_tokens(), cancelled);
                          const auto generate_end = Clock::now();

                          // 5. 把生成出来的所有 Token 塞进 Reply 返回
                          for (const auto token : result.tokens)
                              reply->add_tokens(token);
                          reply->set_finish_reason(result.reason);
                          reply->set_decode_forward_calls(result.decode_calls);
                          reply->set_prompt_forward_tokens(0); // Decode 阶段不需要前向 prompt
                          // 6. 回传分段耗时：快照导入 与 生成
                          reply->set_import_us(to_us(import_end - import_begin));
                          reply->set_generate_us(to_us(generate_end - import_end));
                          reply->set_request_id(request->request_id());
                      });
        }

    private:
        Model model_;
        std::mutex slot_;
    };

}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 5 || std::string(argv[3]) != "-i")
            throw std::invalid_argument("Usage: llm_rpc_inference_worker prefill|decode MODEL -i CONFIG");

        const std::string role = argv[1]; // "prefill" 或 "decode"
        char *config_args[] = {argv[0], argv[3], argv[4], nullptr};
        RpcApplication::Init(3, config_args); // 初始化 RPC 框架（加载配置）

        RpcProvider provider;
        if (role == "prefill")
            provider.NotifyService(new PrefillWorker(argv[2]));
        else if (role == "decode")
            provider.NotifyService(new DecodeWorker(argv[2]));
        else
            throw std::invalid_argument("unknown inference worker role");

        // 启动 RPC 服务，设置 2 个 IO 线程
        provider.Run(2);
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
