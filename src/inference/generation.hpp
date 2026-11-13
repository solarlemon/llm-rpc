#pragma once
#include "model.hpp"
#include <stdexcept>

namespace llm_rpc
{
    namespace inference
    {
        inline void validate_budget(std::size_t positions, std::size_t capacity, int maximum)
        {
            if (!positions || positions > capacity || maximum < 0 ||
                (maximum > 0 && static_cast<std::size_t>(maximum - 1) > capacity - positions))
                throw std::invalid_argument("prompt and generation exceed model context");
        }

        struct Generation
        {
            std::vector<TokenId> tokens;
            Logits last_decode_logits;
            const char *reason;
            int decode_calls = 0;
        };

        inline Generation generate(const Model &model, KVCache &cache, TokenId next,
                                   int maximum, const Cancel &cancelled = {})
        {
            Generation result{{}, {}, maximum ? "length" : "prefill_only", 0};
            for (int i = 0; i < maximum; ++i)
            {
                // 出于资源的考虑，限制最多生成 maximum 个 token，或遇到 EOS/取消时提前结束
                if (model.is_eos(next)) // 遇到停止符号，提前结束
                {
                    result.reason = "eos";
                    break;
                }
                if (cancelled && cancelled())
                    // 超时熔断
                    throw std::runtime_error("inference deadline exceeded");
                // 核心：把当前词喂给模型，结合之前的 KV Cache，算出下一个词的概率分布
                // 包括前一轮计算的 KV Cache，例如 prompt "你好" 的 KV Cache，和当前 token "世"
                result.tokens.push_back(next);
                if (i + 1 < maximum)
                {
                    // 核心：把当前词喂给模型，结合之前的 KV Cache，算出下一个词的概率分布
                    result.last_decode_logits = model.decode(next, cache, cancelled);
                    ++result.decode_calls;
                    next = Model::greedy(result.last_decode_logits);
                }
            }
            return result;
        }

    }
}
