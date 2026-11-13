#include "inference_rpc.pb.h"
#include "snapshot.hpp"
#include "generation.hpp"
#include "rpc_application.h"
#include "rpc_provider.h"
#include "rpc_transport.h"

#include <iostream>
#include <mutex>

namespace
{
    using namespace llm_rpc;
    using namespace llm_rpc::inference;

    // 通用工作流包装器：处理互斥、超时检测、异常捕获、回调
    template <class Reply, class Work>
    void RunWorker(std::mutex &slot, std::uint32_t timeout, Reply *reply,
                   google::protobuf::Closure *done, Work work)
    {
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

                // 3. 计算截止时间点，并定义取消回调
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::milliseconds(timeout);
                const Cancel cancelled = [deadline]
                {
                    return std::chrono::steady_clock::now() >= deadline;
                };

                // 4. 执行真正的推理工作
                work(cancelled);

                // 5. 如果推理结束后发现超时了，也视为失败
                if (cancelled())
                    throw std::runtime_error("inference deadline exceeded");
            }
            catch (const std::exception &error)
            {
                // 6. 捕获任何异常，清空 Reply，并把错误信息返回给客户端
                reply->Clear();
                reply->set_error(error.what());
            }
        } // 释放锁，允许下一个请求进入

        // 7. 执行 RPC 回调，把 Reply 发送回客户端
        done->Run();
    }

    // Prefill 工作节点：计算 KV Cache 并打包
    class PrefillWorker final : public PrefillService
    {
    public:
        explicit PrefillWorker(const std::string &path) : model_(path) {}

        void RunPrefill(google::protobuf::RpcController *, const PrefillRequest *request,
                        PrefillReply *reply, google::protobuf::Closure *done) override
        {
            RunWorker(slot_, request->timeout_ms(), reply, done, [&](const Cancel &cancelled)
                      {
                // 1. 校验 Prompt 长度
                if (request->prompt_size() <= 0 || request->prompt_size() > model_.config().capacity)
                    throw std::invalid_argument("invalid prompt length");
                
                // 2. 把 Protobuf 的数组转成 C++ 的 vector
                const std::vector<TokenId> prompt(request->prompt().begin(), request->prompt().end());
                
                // 3. 执行 Prefill 前向传播（注意 chunk_size 传了 8，代表分块预填充，每次处理 8 个 token 以防 OOM）
                KVCache cache(model_.cache_shape());
                const auto logits = model_.prefill(prompt, cache, 8, cancelled);
                
                // 4. 贪心算法采样出首 Token
                // 例如用户输入的 prompt 是 "Hello", 模型输出的 logits 是一个长度为 vocab_size 的向量，表示下一个 token 的概率分布。
                // 我们使用贪心算法选择概率最高的 token 作为首个生成 token。
                const auto first_token = Model::greedy(logits);
                
                // 5. 把整个 KV Cache 序列化成二进制快照（限制最大包长 rpc_max_frame - 4096）
                // 模型对象：获取模型指纹，确保DecodeWorker能正确加载相同模型；
                // KVCache对象：包含所有层的key/value缓存；first_token：首个生成token；request_id：用于请求追踪。
                // 预留 4096 字节给 RPC 协议头和其他字段，避免超过最大帧长度。
                reply->set_snapshot(KVSnapshot::encode(model_, cache, first_token,
                                                      request->request_id(), rpc_max_frame - 4096));
                reply->set_request_id(request->request_id()); });
        }

    private:
        Model model_;
        std::mutex slot_; // 原子锁，保证同一时刻只有一个请求能占用模型
    };

    // Decode 工作节点：接收快照并连续生成
    class DecodeWorker final : public DecodeService
    {
    public:
        explicit DecodeWorker(const std::string &path) : model_(path) {}

        void RunDecode(google::protobuf::RpcController *, const DecodeRequest *request,
                       DecodeReply *reply, google::protobuf::Closure *done) override
        {
            RunWorker(slot_, request->timeout_ms(), reply, done, [&](const Cancel &cancelled)
                      {
                // 1. 从 RPC 包里解码出 KV Cache 快照
                auto state = KVSnapshot::decode(request->snapshot(), model_);
                
                // 2. 校验 Request ID（防止路由错乱）
                if (state.request_id != request->request_id())
                    throw std::invalid_argument("snapshot request ID mismatch");
                
                // 3. 校验生成长度预算
                if (request->max_tokens() == 0 || request->max_tokens() > model_.cache_shape().capacity + 1)
                    throw std::invalid_argument("invalid generation limit");
                validate_budget(state.cache->valid_len(), model_.cache_shape().capacity, request->max_tokens());
                
                // 4. 开始连续生成（复用 generation.hpp 的 generate 逻辑）
                const auto result = generate(model_, *state.cache, state.first_token,
                                             request->max_tokens(), cancelled);
                
                // 5. 把生成出来的所有 Token 塞进 Reply 返回
                for (const auto token : result.tokens)
                    reply->add_tokens(token);
                reply->set_finish_reason(result.reason);
                reply->set_decode_forward_calls(result.decode_calls);
                reply->set_prompt_forward_tokens(0); // Decode 阶段不需要前向 prompt
                reply->set_request_id(request->request_id()); });
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