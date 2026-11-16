#include "inference_rpc.pb.h"
#include "rpc_application.h"
#include "rpc_channel.h"
#include "rpc_controller.h"
#include "qwen_tokenizer.hpp"
#include <memory>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace
{
    using namespace llm_rpc;
    using namespace llm_rpc::inference;
    using Clock = std::chrono::steady_clock;

    // 严格的参数解析：只允许纯数字，且不能超过最大值
    std::uint32_t parse(const std::string &text, std::uint32_t maximum)
    {
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("expected a nonnegative integer");
        const auto value = std::stoull(text);
        if (value > maximum)
            throw std::invalid_argument("integer outside supported range");
        return static_cast<std::uint32_t>(value);
    }

    // 计算距离截止时间还剩多少毫秒，如果超时则直接抛异常
    std::chrono::milliseconds remaining(Clock::time_point deadline)
    {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
        if (ms.count() <= 0)
            throw std::runtime_error("inference deadline exceeded");
        return ms;
    }

    // 统一检查 RPC 调用状态和业务层错误
    void check(const RpcController &controller, const std::string &error)
    {
        if (controller.Failed())
            throw std::runtime_error(controller.ErrorText());
        if (!error.empty())
            throw std::runtime_error(error);
    }
}

int main(int argc, char **argv)
{
    try
    {
        // 1. 解析命令行参数：配置文件、Prompt ID列表、最大生成数、超时时间、Request ID
        const bool chat = argc == 9 && std::string(argv[3]) == "--chat";
        if ((argc != 7 && !chat) || std::string(argv[1]) != "-i")
            throw std::invalid_argument(
                "Usage: llm_rpc_inference_client -i CONFIG IDS MAX_TOKENS TIMEOUT_MS REQUEST_ID\n"
                "       llm_rpc_inference_client -i CONFIG --chat TOKENIZER_DIR QUESTION MAX_TOKENS TIMEOUT_MS REQUEST_ID");

        char *config_args[] = {argv[0], argv[1], argv[2], nullptr};
        RpcApplication::Init(3, config_args); // 初始化 RPC 框架（从配置文件读取 ZK 地址等）

        const auto maximum = parse(argv[argc - 3], 65536);
        const auto timeout = parse(argv[argc - 2], 60000);
        const auto id = parse(argv[argc - 1], std::numeric_limits<std::uint32_t>::max());
        if (!maximum || !timeout)
            throw std::invalid_argument("generation limit and timeout must be positive");

        // 2. prompt 只构建一次：重复请求时复用，避免反复解析与分词
        std::unique_ptr<QwenTokenizer> tokenizer;
        std::vector<std::int32_t> prompt_ids;
        if (chat)
        {
            tokenizer = std::make_unique<QwenTokenizer>(argv[4]);
            prompt_ids = tokenizer->prompt(argv[5]);
            if (prompt_ids.size() + maximum - 1 > 256)
                throw std::invalid_argument("Qwen prompt and output exceed 256-token context");
        }
        else
        {
            const std::string ids = argv[3];
            for (std::size_t begin = 0;;)
            {
                const auto end = ids.find(',', begin);
                prompt_ids.push_back(parse(ids.substr(begin, end - begin), std::numeric_limits<int>::max()));
                if (end == std::string::npos)
                    break;
                begin = end + 1;
            }
        }

        // 开发期校验：把 chat 模板的分词结果导出，便于与 HuggingFace 的
        // apply_chat_template 逐 id 对照（导出后直接退出，不发起 RPC）
        if (const char *dump_path = std::getenv("LLM_RPC_DUMP_PROMPT"))
        {
            std::ofstream dump(dump_path);
            if (!dump)
                throw std::runtime_error("cannot open prompt dump file");
            for (std::size_t i = 0; i < prompt_ids.size(); ++i)
                dump << (i ? "," : "") << prompt_ids[i];
            dump << '\n';
            return 0;
        }

        // LLM_RPC_REPEAT=N：同一进程内连续发起 N 次请求。
        // 把 ZooKeeper 会话建立（约 36 ms）与服务发现的开销从单次请求里摊掉，
        // 基准脚本的"热态"就是用这个模式测量的。
        const char *repeat_env = std::getenv("LLM_RPC_REPEAT");
        const int repeats = repeat_env ? std::max(1, std::atoi(repeat_env)) : 1;

        for (int iteration = 0; iteration < repeats; ++iteration)
        {
        const auto started_at = Clock::now(); // 分段计时起点

        PrefillRequest prefill;
        for (const auto token : prompt_ids)
            prefill.add_prompt(token);
        prefill.set_request_id(id);
        const auto tokenized_at = Clock::now();

        // 3. 计算全局超时 Deadline，并计算 Prefill 阶段能用的时间
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
        const auto prefill_budget = remaining(deadline);
        prefill.set_timeout_ms(prefill_budget.count());

        // ★ 关键设计：直接创建带超时时间的 Channel，实现 RPC 层的超时控制
        RpcChannel prefill_channel(false, prefill_budget);
        // 创建 Prefill 服务的 Stub，Stub中已经绑定了对应的服务
        PrefillService::Stub prefill_stub(&prefill_channel);
        PrefillReply snapshot;
        RpcController controller;

        // 4. 远程调用 Prefill 服务（获取 KV Cache 快照）
        prefill_stub.RunPrefill(&controller, &prefill, &snapshot, nullptr);
        const auto prefill_done = Clock::now();
        check(controller, snapshot.error());

        if (snapshot.request_id() != id || snapshot.snapshot().empty())
            throw std::runtime_error("invalid Prefill reply");

        const auto snapshot_bytes = snapshot.snapshot().size();

        // 5. 构建 Decode 请求
        DecodeRequest decode;
        decode.set_request_id(id);
        decode.set_max_tokens(maximum);

        // ★ 极致优化：使用 swap 实现 std::string 的零拷贝转移
        // 将刚才收到的 snapshot 里的几十 MB 的 KV Cache 字符串，直接移交给 decode 请求，避免内存复制。
        decode.mutable_snapshot()->swap(*snapshot.mutable_snapshot());

        // 计算 Decode 阶段还能剩多少时间
        const auto decode_budget = remaining(deadline);
        decode.set_timeout_ms(decode_budget.count());

        RpcChannel decode_channel(false, decode_budget);
        DecodeService::Stub decode_stub(&decode_channel);
        DecodeReply reply;
        controller.Reset(); // 控制器复用前必须重置状态

        // 6. 远程调用 Decode 服务（带着快照去生成文本）
        decode_stub.RunDecode(&controller, &decode, &reply, nullptr);
        const auto decode_done = Clock::now();
        check(controller, reply.error());

        remaining(deadline); // 最终再校验一次是否超时
        if (reply.request_id() != id)
            throw std::runtime_error("invalid Decode reply ID");

        // 7. 汇总分段耗时：
        //    客户端侧——分词 / Prefill RPC / Decode RPC；
        //    服务端回传——prefill 计算、快照编码、快照导入、生成。
        const auto us = [](Clock::duration span)
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(span).count();
        };
        std::ostringstream metrics;
        metrics << "\"iteration\":" << iteration
                << ",\"request_id\":" << id
                << ",\"snapshot_bytes\":" << snapshot_bytes
                << ",\"prompt_forward_tokens\":" << reply.prompt_forward_tokens()
                << ",\"decode_forward_calls\":" << reply.decode_forward_calls()
                << ",\"tokenize_us\":" << us(tokenized_at - started_at)
                << ",\"prefill_rpc_us\":" << us(prefill_done - tokenized_at)
                << ",\"prefill_us\":" << snapshot.prefill_us()
                << ",\"encode_us\":" << snapshot.encode_us()
                << ",\"decode_rpc_us\":" << us(decode_done - prefill_done)
                << ",\"import_us\":" << reply.import_us()
                << ",\"generate_us\":" << reply.generate_us()
                << ",\"total_us\":" << us(decode_done - started_at);

        if (chat)
        {
            const std::vector<std::int32_t> tokens(reply.tokens().begin(), reply.tokens().end());
            std::cout << tokenizer->decode(tokens) << '\n';
            // 问答模式下 stdout 只放答案，指标写到 stderr 供基准脚本抓取
            std::cerr << "BENCH {" << metrics.str() << "}\n";
            return 0;
        }

        // 8. 打印标准的 JSON 格式输出，方便外部监控或前端解析
        std::cout << "{" << metrics.str() << ",\"output_ids\":[";
        for (int i = 0; i < reply.tokens_size(); ++i)
            std::cout << (i ? "," : "") << reply.tokens(i);
        std::cout << "],\"finish_reason\":\"" << reply.finish_reason() << "\"}\n";
        } // for (iteration)
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}