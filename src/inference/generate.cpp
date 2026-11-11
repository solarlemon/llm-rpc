#include "generation.hpp"
#include "snapshot.hpp"
#include "tokenizer.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
    using namespace llm_rpc::inference;

    // 1. 基础参数解析工具：把字符串转成整数，并严格校验是否有多余字符
    int parse_integer(const std::string &value)
    {
        std::size_t consumed = 0;
        const int result = std::stoi(value, &consumed);
        if (consumed != value.size())
            throw std::invalid_argument("invalid integer: " + value);
        return result;
    }

    // 2. 解析 Request ID：必须全数字，且不能超过 uint32 上限
    std::uint32_t parse_request_id(const std::string &value)
    {
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("invalid request ID");
        const auto id = std::stoull(value);
        if (id > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("request ID exceeds uint32");
        return static_cast<std::uint32_t>(id);
    }

    // 3. 解析逗号分隔的 Token ID 串（如 "1,23,45,67" 转为 vector）
    std::vector<TokenId> parse_tokens(const std::string &csv)
    {
        std::vector<TokenId> ids;
        for (std::size_t begin = 0;;)
        {
            const auto end = csv.find(',', begin);
            ids.push_back(parse_integer(csv.substr(begin, end - begin)));
            if (end == std::string::npos)
                return ids;
            begin = end + 1;
        }
    }

    // 4. 打印数组，输出成 JSON 格式的数组，如 [1, 2, 3]
    template <class T>
    void print_array(const std::vector<T> &values)
    {
        std::cout << '[';
        for (std::size_t i = 0; i < values.size(); ++i)
            std::cout << (i ? "," : "") << values[i];
        std::cout << ']';
    }

    // 5. 打印生成结果（附带结束原因）
    void print_generation(const Generation &result)
    {
        std::cout << ",\"output_ids\":";
        print_array(result.tokens);
        std::cout << ",\"finish_reason\":\"" << result.reason << "\"}\n";
    }
}

int main(int argc, char **argv)
{
    using namespace llm_rpc::inference;
    try
    {
        // ================= 解析模式 =================
        const std::string mode = argc > 2 ? argv[2] : "";
        const bool token_mode = argc == 5 && mode == "--tokens";    // 直接输入 Token ID
        const bool text_mode = argc == 6 && mode == "--text";       // 输入文本，内部调用 Tokenizer
        const bool prefill_mode = argc == 6 && mode == "--prefill"; // 只做 prefill 并生成快照
        const bool decode_mode = argc == 5 && mode == "--decode";   // 加载快照并解码

        if (!token_mode && !text_mode && !prefill_mode && !decode_mode)
            throw std::invalid_argument(
                "Usage: llm_rpc_generate MODEL --tokens IDS MAX_NEW_TOKENS\n"
                "       llm_rpc_generate MODEL --text PROMPT TOKENIZER MAX_NEW_TOKENS\n"
                "       llm_rpc_generate MODEL --prefill IDS SNAPSHOT REQUEST_ID\n"
                "       llm_rpc_generate MODEL --decode SNAPSHOT MAX_NEW_TOKENS");

        // 加载模型权重
        const Model model(argv[1]);
        const auto capacity = model.cache_shape().capacity;
        // 设置输出精度，保证浮点数无损输出
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);

        // ================= 模式 4: Decode（断点续传解码） =================
        if (decode_mode)
        {
            const int maximum = parse_integer(argv[4]);
            // 从磁盘恢复 KV Cache 快照
            auto state = KVSnapshot::load(argv[3], model);
            const auto positions = state.cache->valid_len();
            validate_budget(positions, capacity, maximum);

            // 基于恢复的 Cache 生成后续 Token
            const auto result = generate(model, *state.cache, state.first_token, maximum);

            // 打印 JSON 结果
            std::cout << "{\"request_id\":" << state.request_id
                      << ",\"restored_prompt_tokens\":" << positions
                      << ",\"prompt_forward_tokens\":0"
                      << ",\"decode_forward_calls\":" << result.decode_calls
                      << ",\"last_decode_logits\":";
            print_array(result.last_decode_logits);
            print_generation(result);
            return 0;
        }

        // ================= 前处理 =================
        const int maximum = prefill_mode ? 0 : parse_integer(argv[argc - 1]);

        // 如果传了文本，用 Tokenizer 编码；如果直接传了 Token 串，直接解析
        const auto prompt = text_mode
                                ? Tokenizer(argv[4], model.config().vocab).encode(argv[3])
                                : parse_tokens(argv[3]);

        // 检查提示词长度 + 要生成的 Token 数，是否超出了模型物理 KV Cache 的极限（capacity）
        validate_budget(prompt.size(), capacity, maximum);

        // ================= 模式 1 & 2 & 3: Prefill =================
        KVCache cache(model.cache_shape());
        // 传入所有 Prompt Token，计算最后一次的 logits
        const auto logits = model.prefill(prompt, cache);
        // 拿到预测的第一个 Token
        const auto first = Model::greedy(logits);

        // ================= 模式 3: Prefill (只计算并保存快照) =================
        if (prefill_mode)
        {
            const auto request = parse_request_id(argv[5]);
            // 把当前的 KV Cache 状态序列化保存到文件
            KVSnapshot::save(argv[4], model, cache, first, request);
            std::cout << "{\"request_id\":" << request
                      << ",\"prompt_forward_tokens\":" << cache.valid_len()
                      << ",\"kv_payload_bytes\":" << cache.valid_bytes()
                      << ",\"first_token_id\":" << first << "}\n";
            return 0;
        }

        // ================= 模式 1 & 2: 生成并打印 =================
        // 基于 prefill 的 cache 和首 token，继续自回归生成
        const auto result = generate(model, cache, first, maximum);
        std::cout << "{\"prompt_ids\":";
        print_array(prompt);
        std::cout << ",\"prefill_logits\":";
        print_array(logits);
        print_generation(result);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}