#include "model.hpp"
#include "safetensors.hh"

#include <cstring>
#include <stdexcept>

namespace llm_rpc
{
    namespace inference
    {
        void Model::load_qwen(const std::string &checkpoint)
        {
            safetensors::safetensors_t weights;
            std::string error;
            if (!safetensors::mmap_from_file(checkpoint, &weights, nullptr, &error) ||
                !safetensors::validate_data_offsets(weights, error))
                throw std::runtime_error("invalid Qwen safetensors: " + error);

            // Fixed profile for the official Qwen2.5-0.5B-Instruct checkpoint.
            config_ = {896, 4864, 24, 14, 2, 151936, 256};
            qwen_ = true;
            constexpr std::size_t dim = 896, hidden = 4864, kv = 128;
            auto tensor = [&](const std::string &name, std::vector<std::size_t> shape)
            {
                safetensors::tensor_t info{};
                if (!weights.tensors.at(name, &info) || info.shape != shape ||
                    info.dtype != safetensors::kBFLOAT16)
                    throw std::runtime_error("missing or incompatible BF16 tensor: " + name);
                std::size_t count = 1;
                for (const auto n : shape)
                    count *= n;
                const auto begin = info.data_offsets[0], end = info.data_offsets[1];
                if (end < begin || end > weights.databuffer_size || end - begin != count * 2)
                    throw std::runtime_error("invalid tensor extent: " + name);
                const auto *bytes = weights.databuffer_addr + begin;
                std::vector<float> result(count);
                for (std::size_t i = 0; i < count; ++i)
                {
                    const std::uint32_t bits = (std::uint32_t(bytes[2 * i]) << 16) |
                                               (std::uint32_t(bytes[2 * i + 1]) << 24);
                    if ((bits & 0x7f800000U) == 0x7f800000U)
                        throw std::runtime_error("nonfinite tensor: " + name);
                    std::memcpy(&result[i], &bits, sizeof(bits));
                }
                return result;
            };
            if (weights.tensors.count("lm_head.weight"))
                throw std::runtime_error("expected official tied-embedding Qwen checkpoint");
            embedding_ = tensor("model.embed_tokens.weight", {151936, dim});
            final_norm_ = tensor("model.norm.weight", {dim});
            layers_.resize(config_.layers);
            for (std::size_t i = 0; i < layers_.size(); ++i)
            {
                auto &layer = layers_[i];
                const auto base = "model.layers." + std::to_string(i) + ".";
                layer.attention_norm = tensor(base + "input_layernorm.weight", {dim});
                layer.ffn_norm = tensor(base + "post_attention_layernorm.weight", {dim});
                layer.q = tensor(base + "self_attn.q_proj.weight", {dim, dim});
                layer.k = tensor(base + "self_attn.k_proj.weight", {kv, dim});
                layer.v = tensor(base + "self_attn.v_proj.weight", {kv, dim});
                layer.q_bias = tensor(base + "self_attn.q_proj.bias", {dim});
                layer.k_bias = tensor(base + "self_attn.k_proj.bias", {kv});
                layer.v_bias = tensor(base + "self_attn.v_proj.bias", {kv});
                layer.projection = tensor(base + "self_attn.o_proj.weight", {dim, dim});
                layer.gate = tensor(base + "mlp.gate_proj.weight", {hidden, dim});
                layer.down = tensor(base + "mlp.down_proj.weight", {dim, hidden});
                layer.up = tensor(base + "mlp.up_proj.weight", {hidden, dim});
            }
            // Include a profile tag so different forward/KV semantics cannot share a snapshot.
            fingerprint_ = 14695981039346656037ULL;
            for (const char byte : std::string("qwen2.5-0.5b-fp32-kv256-v1"))
                fingerprint_ = (fingerprint_ ^ static_cast<unsigned char>(byte)) * 1099511628211ULL;
            for (std::size_t i = 0; i < weights.mmap_size; ++i)
                fingerprint_ = (fingerprint_ ^ weights.mmap_addr[i]) * 1099511628211ULL;
        }
    }
}
