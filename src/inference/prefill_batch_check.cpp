#include "model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace llm_rpc::inference;

int main(int argc, char **argv)
{
    try
    {
        if (argc != 3)
            throw std::invalid_argument("Usage: llm_rpc_prefill_check MODEL PROMPT_TOKENS");
        const Model model(argv[1]);
        const int count = std::stoi(argv[2]);
        if (count < 1 || count > model.config().capacity)
            throw std::invalid_argument("prompt length outside model context");

        std::vector<TokenId> ids;
        for (int i = 0; i < count; ++i)
            ids.push_back(model.uses_qwen() ? 1000 + i : 3 + i);

        KVCache scalar(model.cache_shape()), batched(model.cache_shape());
        const auto scalar_begin = std::chrono::steady_clock::now();
        Logits expected = model.prefill({ids.front()}, scalar, 1);
        for (std::size_t i = 1; i < ids.size(); ++i)
            expected = model.decode(ids[i], scalar);
        const auto batch_begin = std::chrono::steady_clock::now();
        const auto actual = model.prefill(ids, batched, 8);
        const auto batch_end = std::chrono::steady_clock::now();

        if (scalar.valid_len() != batched.valid_len() || actual.size() != expected.size())
            throw std::runtime_error("prefill output shape mismatch");
        float max_logits = 0, max_kv = 0;
        for (std::size_t i = 0; i < actual.size(); ++i)
            max_logits = std::max(max_logits, std::abs(actual[i] - expected[i]));
        const auto shape = model.cache_shape();
        for (std::size_t layer = 0; layer < shape.layers; ++layer)
            for (std::size_t pos = 0; pos < ids.size(); ++pos)
                for (std::size_t head = 0; head < shape.kv_heads; ++head)
                    for (std::size_t dim = 0; dim < shape.head_dim; ++dim)
                    {
                        max_kv = std::max(max_kv, std::abs(
                            scalar.key(layer, pos, head, dim) -
                            batched.key(layer, pos, head, dim)));
                        max_kv = std::max(max_kv, std::abs(
                            scalar.value(layer, pos, head, dim) -
                            batched.value(layer, pos, head, dim)));
                    }
        const auto elapsed_ms = [](std::chrono::steady_clock::duration duration)
        {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        std::cout << "tokens=" << count << " max_logits=" << max_logits
                  << " max_kv=" << max_kv
                  << " scalar_ms=" << elapsed_ms(batch_begin - scalar_begin)
                  << " batch_ms=" << elapsed_ms(batch_end - batch_begin) << '\n';
        return max_logits <= 1e-3f && max_kv <= 1e-4f ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
