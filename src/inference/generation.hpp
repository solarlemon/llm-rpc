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
                if (next == 2) // stories260K EOS.
                {
                    result.reason = "eos";
                    break;
                }
                if (cancelled && cancelled())
                    throw std::runtime_error("inference deadline exceeded");
                result.tokens.push_back(next);
                if (i + 1 < maximum)
                {
                    result.last_decode_logits = model.decode(next, cache, cancelled);
                    ++result.decode_calls;
                    next = Model::greedy(result.last_decode_logits);
                }
            }
            return result;
        }

    }
}
