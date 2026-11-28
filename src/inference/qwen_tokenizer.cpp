#include "qwen_tokenizer.hpp"
#include "json.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/ustring.h>

namespace llm_rpc
{
    namespace
    {
        using Json = nlohmann::json;        // 依赖 nlohmann/json 库解析 tokenizer.json
        constexpr int start_token = 151644; // <|im_start|> 的 ID
        constexpr int end_token = 151645;   // <|im_end|> 的 ID
        constexpr int vocab_size = 151936;  // Qwen2.5-0.5B 的总词表大小
        constexpr char system_prompt[] = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

        // 读 JSON 文件
        Json load_json(const std::string &file) { std::ifstream input(file); if (!input) throw std::runtime_error("cannot open tokenizer JSON: " + file); return Json::parse(input); }

        // ICU 错误检查（U_FAILURE 是 ICU 提供的宏）
        void check_icu(UErrorCode status)
        {
            if (U_FAILURE(status))
                throw std::runtime_error(u_errorName(status));
        }

        // 将两个 32 位 Token ID 压缩成一个 64 位整数，作为 HashMap 的 Key
        std::uint64_t pair_key(int left, int right)
        {
            return (std::uint64_t(left) << 32) | std::uint32_t(right);
        }
    }

    struct QwenTokenizer::Data
    {
        struct Merge
        {
            std::size_t rank;
            int id;
        };

        // 核心成员：
        std::array<int, 256> bytes{};                              // 256个基础字节 到 Token ID 的映射
        std::vector<std::string> decoded{vocab_size};              // Token ID 到 字符串(UTF-8) 的逆向映射
        std::vector<bool> regular = std::vector<bool>(vocab_size); // 标记该 ID 是否是常规词
        std::unordered_map<std::uint64_t, Merge> merges;           // BPE 合并规则表（Key: left/right 组合，Value: 优先级和合并后的ID）
        std::unique_ptr<icu::RegexPattern> pattern;                // ICU 正则匹配器（用于切分文本）

        void encode(const std::string &text, std::vector<std::int32_t> &output) const
        {
            UErrorCode status = U_ZERO_ERROR;

            // 1. NFC Unicode 归一化（比如把带有声调的字符标准化）
            const auto *nfc = icu::Normalizer2::getNFCInstance(status);
            check_icu(status);
            const auto normalized = nfc->normalize(icu::UnicodeString::fromUTF8(text), status);
            check_icu(status);

            // 2. 正则预切分（Pre-tokenization）
            // Qwen 会把文本切成一个个词（比如按空格、标点、数字切）
            std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(normalized, status));
            check_icu(status);
            int32_t offset = 0;

            while (matcher->find(status))
            {
                if (matcher->start(status) != offset)
                    throw std::runtime_error("unmatched Qwen input");
                offset = matcher->end(status);

                std::string piece;
                matcher->group(status).toUTF8String(piece); // 拿到切分好的一个片段
                check_icu(status);

                // 3. 将片段转换成 Byte 级别的 ID 序列
                std::vector<int> ids;
                for (unsigned char byte : piece)
                    ids.push_back(bytes[byte]);

                // 4. 核心 BPE 合并循环
                while (ids.size() > 1)
                {
                    auto best = std::numeric_limits<std::size_t>::max();
                    std::uint64_t selected = 0;
                    int merged = 0;

                    // 遍历所有相邻的两个 ID，在 merges 表中找优先级最高（rank 最小）的合并规则
                    for (std::size_t i = 1; i < ids.size(); ++i)
                    {
                        const auto key = pair_key(ids[i - 1], ids[i]);
                        const auto it = merges.find(key);
                        if (it != merges.end() && it->second.rank < best)
                        {
                            best = it->second.rank;
                            selected = key;
                            merged = it->second.id;
                        }
                    }
                    if (best == std::numeric_limits<std::size_t>::max())
                        break; // 没有可合并的，退出

                    // 执行合并：把相邻的两个 ID 替换成合并后的 ID
                    std::vector<int> next;
                    for (std::size_t i = 0; i < ids.size(); ++i)
                    {
                        if (i + 1 < ids.size() && pair_key(ids[i], ids[i + 1]) == selected)
                        {
                            next.push_back(merged);
                            ++i; // 跳过下一个
                        }
                        else
                            next.push_back(ids[i]);
                    }
                    ids.swap(next);
                }
                output.insert(output.end(), ids.begin(), ids.end());
            }
            check_icu(status);
            if (offset != normalized.length())
                throw std::runtime_error("incomplete Qwen tokenization");
        }
    };

    QwenTokenizer::QwenTokenizer(const std::string &directory)
    {
        // 1. 严格的格式校验（防止加载了错误的模型）
        const auto config = load_json(directory + "/tokenizer_config.json");
        const auto format = config.at("chat_template").get<std::string>();
        // 确保 Chat Template 里有 Qwen 特有的 system_prompt 和 <|im_start|>assistant
        if (format.find(system_prompt) == std::string::npos || format.find("<|im_start|>assistant") == std::string::npos)
            throw std::runtime_error("unsupported Qwen chat template");

        const auto json = load_json(directory + "/tokenizer.json");
        const auto &model = json.at("model");
        const auto &vocab = model.at("vocab");
        const auto &split = json.at("pre_tokenizer").at("pretokenizers");

        // 校验必须是 BPE、NFC 归一化、ByteLevel 解码、两阶段预切分
        if (model.at("type") != "BPE" || json.at("normalizer").at("type") != "NFC" ||
            json.at("decoder").at("type") != "ByteLevel" || split.size() != 2 ||
            split.at(0).at("behavior") != "Isolated" || split.at(0).at("invert") != false ||
            split.at(1).at("type") != "ByteLevel" || split.at(1).at("use_regex") != false ||
            split.at(1).at("add_prefix_space") != false)
            throw std::runtime_error("unsupported Qwen tokenizer format");

        bool has_start = false, has_end = false;
        for (const auto &token : json.at("added_tokens"))
        {
            has_start |= token.at("id") == start_token && token.at("content") == "<|im_start|>";
            has_end |= token.at("id") == end_token && token.at("content") == "<|im_end|>";
        }
        if (!has_start || !has_end || vocab.size() != 151643)
            throw std::runtime_error("unsupported Qwen vocabulary");

        auto data = std::make_shared<Data>();
        UErrorCode status = U_ZERO_ERROR;

        // 2. 编译正则表达式（用于将输入文本切分成基础的词块）
        data->pattern.reset(icu::RegexPattern::compile(
            icu::UnicodeString::fromUTF8(split.at(0).at("pattern").at("Regex").get<std::string>()), 0, status));
        check_icu(status);

        // 3. 构建 Byte 级别的映射表（Qwen 使用了类似 GPT-2 的 Byte-level BPE）
        // 将 256 个字节映射到特定的 Unicode 字符上，再把 Unicode 映射到 Token ID
        std::unordered_map<UChar32, unsigned char> inverse;
        int extra = 256;
        for (int byte = 0; byte < 256; ++byte)
        {
            // 可见字符直接用，不可见字符（控制符、特殊字符）往后排
            const bool direct = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255);
            const UChar32 point = direct ? byte : extra++;
            inverse.emplace(point, byte);
            std::string symbol;
            icu::UnicodeString(point).toUTF8String(symbol);
            data->bytes[byte] = vocab.at(symbol).get<int>();
        }

        // 4. 构建逆向解码表（Token ID -> UTF-8 字符串）
        for (auto it = vocab.begin(); it != vocab.end(); ++it)
        {
            const int id = it.value().get<int>();
            if (id < 0 || id >= 151643 || data->regular[id])
                throw std::runtime_error("invalid Qwen token ID");

            const auto symbols = icu::UnicodeString::fromUTF8(it.key());
            for (int32_t i = 0; i < symbols.length();)
            {
                const auto point = symbols.char32At(i);
                data->decoded[id].push_back(static_cast<char>(inverse.at(point)));
                i += U16_LENGTH(point);
            }
            data->regular[id] = true;
        }

        // 5. 加载 BPE 合并规则表（注意：rank 越小，优先级越高）
        std::size_t rank = 0;
        for (const auto &merge : model.at("merges"))
        {
            const auto left = merge.at(0).get<std::string>();
            const auto right = merge.at(1).get<std::string>();
            // 将左词、右词合并为完整的 Key，并把合并后的 ID 和优先级存起来
            data->merges.emplace(pair_key(vocab.at(left).get<int>(), vocab.at(right).get<int>()),
                                 Data::Merge{rank++, vocab.at(left + right).get<int>()});
        }
        data_ = std::move(data);
    }

    // 组装符合 Qwen 官方 Chat Template 格式的完整 Prompt
    std::vector<std::int32_t> QwenTokenizer::prompt(const std::string &question) const
    {
        if (question.empty() || question.size() > 4096)
            throw std::invalid_argument("question must contain 1 to 4096 UTF-8 bytes");

        std::vector<std::int32_t> tokens{start_token}; // <|im_start|>
        data_->encode(std::string("system\n") + system_prompt, tokens);
        tokens.push_back(end_token); // <|im_end|>
        data_->encode("\n", tokens);
        tokens.push_back(start_token); // <|im_start|>
        data_->encode("user\n" + question, tokens);
        tokens.push_back(end_token); // <|im_end|>
        data_->encode("\n", tokens);
        tokens.push_back(start_token); // <|im_start|>
        data_->encode("assistant\n", tokens);
        return tokens;
    }

    // 解码：把生成的 Token ID 列表变回人类能看的字符串
    std::string QwenTokenizer::decode(const std::vector<std::int32_t> &ids) const
    {
        std::string bytes;
        for (const auto id : ids)
        {
            if (id < 0 || id >= vocab_size)
                throw std::runtime_error("invalid Qwen output token");
            // 如果是常规词（不是特殊控制符），就把它的字节拼起来
            if (data_->regular[id])
                bytes += data_->decoded[id];
        }
        // ICU 修复可能被截断的 UTF-8 字节序列，避免输出乱码（替换为 U+FFFD）
        std::string result;
        icu::UnicodeString::fromUTF8(bytes).toUTF8String(result);
        return result;
    }
}
