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

namespace llm_rpc {
namespace {
using Json = nlohmann::json;
constexpr int start_token = 151644;
constexpr int end_token = 151645;
constexpr int vocab_size = 151936;
constexpr char system_prompt[] = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

Json load_json(const std::string &file)
{
    std::ifstream input(file);
    if (!input)
        throw std::runtime_error("cannot open " + file);
    return Json::parse(input);
}

void check_icu(UErrorCode status)
{
    if (U_FAILURE(status))
        throw std::runtime_error(u_errorName(status));
}

std::uint64_t pair_key(int left, int right)
{
    return (std::uint64_t(left) << 32) | std::uint32_t(right);
}
}

struct QwenTokenizer::Data {
    struct Merge { std::size_t rank; int id; };
    std::array<int, 256> bytes{};
    std::vector<std::string> decoded{vocab_size};
    std::vector<bool> regular = std::vector<bool>(vocab_size);
    std::unordered_map<std::uint64_t, Merge> merges;
    std::unique_ptr<icu::RegexPattern> pattern;

    void encode(const std::string &text, std::vector<std::int32_t> &output) const
    {
        UErrorCode status = U_ZERO_ERROR;
        int32_t count = 0;
        u_strFromUTF8(nullptr, 0, &count, text.data(), text.size(), &status);
        if (status == U_BUFFER_OVERFLOW_ERROR)
            status = U_ZERO_ERROR;
        check_icu(status);
        const auto *nfc = icu::Normalizer2::getNFCInstance(status);
        check_icu(status);
        const auto normalized = nfc->normalize(icu::UnicodeString::fromUTF8(text), status);
        check_icu(status);
        std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(normalized, status));
        check_icu(status);
        int32_t offset = 0;
        while (matcher->find(status))
        {
            if (matcher->start(status) != offset)
                throw std::runtime_error("unmatched Qwen input");
            offset = matcher->end(status);
            std::string piece;
            matcher->group(status).toUTF8String(piece);
            check_icu(status);
            std::vector<int> ids;
            for (unsigned char byte : piece)
                ids.push_back(bytes[byte]);
            while (ids.size() > 1)
            {
                auto best = std::numeric_limits<std::size_t>::max();
                std::uint64_t selected = 0;
                int merged = 0;
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
                    break;
                std::vector<int> next;
                for (std::size_t i = 0; i < ids.size(); ++i)
                {
                    if (i + 1 < ids.size() && pair_key(ids[i], ids[i + 1]) == selected)
                    {
                        next.push_back(merged);
                        ++i;
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
    const auto config = load_json(directory + "/tokenizer_config.json");
    const auto format = config.at("chat_template").get<std::string>();
    if (format.find(system_prompt) == std::string::npos ||
        format.find("<|im_start|>assistant") == std::string::npos)
        throw std::runtime_error("unsupported Qwen chat template");
    const auto json = load_json(directory + "/tokenizer.json");
    const auto &model = json.at("model");
    const auto &vocab = model.at("vocab");
    const auto &split = json.at("pre_tokenizer").at("pretokenizers");
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
    data->pattern.reset(icu::RegexPattern::compile(
        icu::UnicodeString::fromUTF8(split.at(0).at("pattern").at("Regex").get<std::string>()), 0, status));
    check_icu(status);

    std::unordered_map<UChar32, unsigned char> inverse;
    int extra = 256;
    for (int byte = 0; byte < 256; ++byte)
    {
        const bool direct = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                            (byte >= 174 && byte <= 255);
        const UChar32 point = direct ? byte : extra++;
        inverse.emplace(point, byte);
        std::string symbol;
        icu::UnicodeString(point).toUTF8String(symbol);
        data->bytes[byte] = vocab.at(symbol).get<int>();
    }
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
    std::size_t rank = 0;
    for (const auto &merge : model.at("merges"))
    {
        const auto left = merge.at(0).get<std::string>();
        const auto right = merge.at(1).get<std::string>();
        data->merges.emplace(pair_key(vocab.at(left).get<int>(), vocab.at(right).get<int>()),
                             Data::Merge{rank++, vocab.at(left + right).get<int>()});
    }
    data_ = std::move(data);
}

std::vector<std::int32_t> QwenTokenizer::prompt(const std::string &question) const
{
    if (question.empty() || question.size() > 4096)
        throw std::invalid_argument("question must contain 1 to 4096 UTF-8 bytes");
    std::vector<std::int32_t> tokens{start_token};
    data_->encode(std::string("system\n") + system_prompt, tokens);
    tokens.push_back(end_token);
    data_->encode("\n", tokens);
    tokens.push_back(start_token);
    data_->encode("user\n" + question, tokens);
    tokens.push_back(end_token);
    data_->encode("\n", tokens);
    tokens.push_back(start_token);
    data_->encode("assistant\n", tokens);
    return tokens;
}

std::string QwenTokenizer::decode(const std::vector<std::int32_t> &ids) const
{
    std::string bytes;
    for (const auto id : ids)
    {
        if (id < 0 || id >= vocab_size)
            throw std::runtime_error("invalid Qwen output token");
        if (data_->regular[id])
            bytes += data_->decoded[id];
    }
    // ICU substitutes U+FFFD for incomplete or malformed UTF-8 output.
    std::string result;
    icu::UnicodeString::fromUTF8(bytes).toUTF8String(result);
    return result;
}
}
