// Layout and math reference: Andrej Karpathy's llama2.c (MIT), see docs/native-inference-m0.md.
#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace llm_rpc
{
    namespace inference
    {
        namespace
        {
            // 编译期强制 IEEE754 FP32：RoPE/softmax 的数值语义依赖它
            static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                          "mini model requires IEEE754 FP32");

            // 取消检查：cancel 为空则跳过；取消时抛 runtime_error
            void check_cancel(const Cancel &cancel)
            {
                if (cancel && cancel())
                    throw std::runtime_error("mini_cancelled");
            }

            // 矩阵-向量乘：w 是 rows × x.size() 的扁平行主序
            std::vector<float> matvec(const std::vector<float> &w, const std::vector<float> &x, int rows)
            {
                std::vector<float> out(rows);
                for (int r = 0; r < rows; ++r)
                {
                    float sum = 0;
                    for (std::size_t c = 0; c < x.size(); ++c)
                        sum += w[r * x.size() + c] * x[c];
                    out[r] = sum;
                }
                return out;
            }

            // 输入按 [token][input_dim] 排列，权重按 [output_dim][input_dim] 排列。
            // 同一权重元素连续用于块内所有 token，避免每个 token 重新遍历整份权重。
            std::vector<float> matmul_block(const std::vector<float> &w,
                                            const std::vector<float> &x,
                                            std::size_t batch, std::size_t input_dim,
                                            int output_dim, const Cancel &cancel)
            {
                std::vector<float> out(batch * static_cast<std::size_t>(output_dim), 0.0f);
                for (int row = 0; row < output_dim; ++row)
                {
                    if ((row & 63) == 0)
                        check_cancel(cancel);
                    for (std::size_t col = 0; col < input_dim; ++col)
                    {
                        const float weight = w[static_cast<std::size_t>(row) * input_dim + col];
                        for (std::size_t token = 0; token < batch; ++token)
                            out[token * output_dim + row] +=
                                weight * x[token * input_dim + col];
                    }
                }
                return out;
            }

            // 每个 token 独立 RMSNorm，仍保持 token-major 排列。
            std::vector<float> norm_block(const std::vector<float> &x,
                                          const std::vector<float> &weight, float epsilon)
            {
                const auto width = weight.size();
                std::vector<float> out(x.size());
                for (std::size_t token = 0; token < x.size() / width; ++token)
                {
                    const auto offset = token * width;
                    float sum = 0;
                    for (std::size_t i = 0; i < width; ++i)
                        sum += x[offset + i] * x[offset + i];
                    const float scale =
                        1.0f / std::sqrt(sum / static_cast<float>(width) + epsilon);
                    for (std::size_t i = 0; i < width; ++i)
                        out[offset + i] = x[offset + i] * scale * weight[i];
                }
                return out;
            }

            // RMSNorm：x * (1/sqrt(mean(x²)+eps)) * weight
            std::vector<float> norm(const std::vector<float> &x, const std::vector<float> &weight, float epsilon)
            {
                float sum = 0;
                for (float v : x)
                    sum += v * v;
                const float scale = 1.0f / std::sqrt(sum / static_cast<float>(x.size()) + epsilon);
                auto out = x;
                for (std::size_t i = 0; i < x.size(); ++i)
                    out[i] = x[i] * scale * weight[i];
                return out;
            }

            // residual 残差原地累加
            void residual(std::vector<float> &x, const std::vector<float> &delta)
            {
                for (std::size_t i = 0; i < x.size(); ++i)
                    x[i] += delta[i];
            }

            // RoPE：两两配对旋转；频率按 (i % head_dim)/head_dim 计算
            void rope(std::vector<float> &x, std::size_t position, int head_dim, bool qwen)
            {
                for (std::size_t head = 0; head < x.size(); head += head_dim)
                    for (int pair = 0; pair < head_dim / 2; ++pair)
                    {
                        const auto left = head + (qwen ? pair : 2 * pair);
                        const auto right = left + (qwen ? head_dim / 2 : 1);
                        const auto frequency = 1.0f / std::pow(qwen ? 1000000.0f : 10000.0f,
                                                               2.0f * pair / head_dim);
                        const auto angle = static_cast<float>(position) * frequency;
                        const auto a = x[left], b = x[right];
                        x[left] = a * std::cos(angle) - b * std::sin(angle);
                        x[right] = a * std::sin(angle) + b * std::cos(angle);
                    }
            }
        }

        // ---------- 构造：加载 legacy llama2.c checkpoint ----------
        Model::Model(const std::string &checkpoint)
        {
            // 端序必须是 little-endian
            const std::uint32_t endian_probe = 1;
            if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1)
                throw std::runtime_error("mini checkpoint requires little-endian host");

            const std::string extension = ".safetensors";
            if (checkpoint.size() >= extension.size() &&
                checkpoint.compare(checkpoint.size() - extension.size(), extension.size(), extension) == 0)
            {
                load_qwen(checkpoint);
                return;
            }

            // 打开并取大小
            std::ifstream input(checkpoint, std::ios::binary | std::ios::ate);
            if (!input)
                throw std::runtime_error("cannot open mini checkpoint");
            const std::streamoff bytes = input.tellg();
            // 至少 28 字节头（因为必须包含 7 个 int32）；上限 256MB；浮点区必须 4 字节对齐
            if (bytes < 28 || bytes > 256 * 1024 * 1024 || (bytes - 28) % 4 != 0)
                throw std::runtime_error("invalid mini checkpoint byte size");

            // 读 7 个 int32 头
            input.seekg(0);
            std::int32_t header[7];
            input.read(reinterpret_cast<char *>(header), sizeof(header));
            if (!input || header[5] == std::numeric_limits<std::int32_t>::min())
                throw std::runtime_error("invalid mini header");

            // header[5] 符号编码：>0 共享 classifier/embedding，<0 不共享
            // 在 Qwen 共享词嵌入的配置下，输入嵌入矩阵和输出分类器矩阵在物理上是同一份参数。这样可以减少参数量。
            const bool shared = header[5] > 0;
            config_ = {header[0], header[1], header[2], header[3], header[4],
                       std::abs(header[5]), header[6]};
            const auto &c = config_;

            // 维度范围与注意力形状校验
            for (int value : {c.dim, c.hidden_dim, c.layers, c.heads, c.kv_heads, c.vocab, c.capacity})
                if (value <= 0 || value > 65536)
                    throw std::runtime_error("unsupported mini dimension");
            if (c.layers > 128 || c.dim % c.heads || c.heads % c.kv_heads || (c.dim / c.heads) % 2)
                throw std::runtime_error("invalid mini attention shape");

            // 一次性读入浮点区
            std::vector<float> data(static_cast<std::size_t>(bytes - 28) / sizeof(float));
            input.read(reinterpret_cast<char *>(data.data()),
                       static_cast<std::streamsize>(data.size() * sizeof(float)));
            if (!input)
                throw std::runtime_error("truncated mini checkpoint");
            if (!std::all_of(data.begin(), data.end(), [](float x)
                             { return std::isfinite(x); }))
                throw std::runtime_error("nonfinite mini weights");

            // Hash the exact checkpoint bytes used to construct this model.
            // FNV-1a 哈希算法，对整个权重文件（包括 28 字节头和所有的浮点数）计算一个 64 位的哈希值。
            fingerprint_ = 14695981039346656037ULL;
            const auto hash_bytes = [this](const void *buffer, std::size_t size)
            {
                const auto *raw = static_cast<const unsigned char *>(buffer);
                for (std::size_t i = 0; i < size; ++i)
                {
                    fingerprint_ ^= raw[i];
                    fingerprint_ *= 1099511628211ULL;
                }
            };
            hash_bytes(header, sizeof(header));
            hash_bytes(data.data(), data.size() * sizeof(float));

            // 按固定顺序切张量的函数
            std::size_t offset = 0;
            auto take = [&](std::size_t count)
            {
                if (count > data.size() - offset) // 减法避免溢出
                    throw std::runtime_error("truncated mini tensor");
                std::vector<float> result(data.begin() + offset, data.begin() + offset + count);
                offset += count;
                return result;
            };
            const std::size_t d = c.dim, h = c.hidden_dim,
                              kd = d * c.kv_heads / c.heads;

            embedding_ = take(static_cast<std::size_t>(c.vocab) * d); // 切分出 embedding
            layers_.resize(c.layers);
            for (auto &l : layers_)
                l.attention_norm = take(d); // 把所有层的 attention_norm 连续切出来
            for (auto &l : layers_)
                l.q = take(d * d);
            for (auto &l : layers_)
                l.k = take(kd * d); // GQA：kd × d
            for (auto &l : layers_)
                l.v = take(kd * d);
            for (auto &l : layers_)
                l.projection = take(d * d);
            for (auto &l : layers_)
                l.ffn_norm = take(d);
            for (auto &l : layers_)
                l.gate = take(h * d);
            for (auto &l : layers_)
                l.down = take(d * h);
            for (auto &l : layers_)
                l.up = take(h * d);
            final_norm_ = take(d);
            // 文件里的两份 RoPE 频率表跳过（前向实时计算）
            take(static_cast<std::size_t>(c.capacity) * (d / c.heads));
            // classifier：共享则复用 embedding_，否则单独读
            if (!shared)
                classifier_ = take(static_cast<std::size_t>(c.vocab) * d);
            if (offset != data.size())
                throw std::runtime_error("unexpected mini checkpoint trailing data");
        }

        // ---------- 由 Config 推导缓存形状 ----------
        // 把 Config 映射成 KVShape。head_dim = dim / heads
        KVShape Model::cache_shape() const noexcept
        {
            return {static_cast<std::size_t>(config_.layers),
                    static_cast<std::size_t>(config_.capacity),
                    static_cast<std::size_t>(config_.kv_heads),
                    static_cast<std::size_t>(config_.dim / config_.heads)};
        }

        // ---------- 缓存形状校验：capacity 允许更小，其余必须精确匹配 ----------
        void Model::validate_cache(const KVCache &cache) const
        {
            const auto expected = cache_shape();
            const auto &actual = cache.shape();
            if (actual.layers != expected.layers || actual.capacity > expected.capacity ||
                actual.kv_heads != expected.kv_heads || actual.head_dim != expected.head_dim)
                throw std::invalid_argument("mini cache/model shape mismatch");
        }

        // ---------- 单 token 完整前向->计算注意力 ----------
        Logits Model::forward_token(TokenId token, KVCache &cache, const Cancel &cancel) const
        {
            validate_cache(cache); // 形状匹配
            if (token < 0 || token >= config_.vocab)
                throw std::out_of_range("mini token out of range");
            const auto position = cache.valid_len(); // 当前 token 的位置
            if (position >= cache.shape().capacity)
                throw std::out_of_range("mini context exhausted");

            // 维度推导
            const int d = config_.dim, head_dim = d / config_.heads;
            const int kv_dim = head_dim * config_.kv_heads,
                      sharing = config_.heads / config_.kv_heads;

            // 嵌入查表：取 token 对应的 d 维向量
            std::vector<float> x(embedding_.begin() + token * d,
                                 embedding_.begin() + (token + 1) * d);
            try
            {
                // 进入多层 Transformer：每层写入 KV，计算注意力 + FFN
                for (std::size_t layer = 0; layer < layers_.size(); ++layer)
                {
                    check_cancel(cancel); // 逐层检查取消
                    const auto &weights = layers_[layer];
                    const float epsilon = qwen_ ? 1e-6f : 1e-5f;

                    // ---- attention 子层 ----
                    auto normalized = norm(x, weights.attention_norm, epsilon);
                    // QKV 投影：q = W_q · normalized（d 维）、k/v = W_k/W_v · normalized（kv_dim 维）
                    auto q = matvec(weights.q, normalized, d);      // d 维
                    auto k = matvec(weights.k, normalized, kv_dim); // kv_dim 维
                    auto v = matvec(weights.v, normalized, kv_dim);
                    if (qwen_)
                    {
                        residual(q, weights.q_bias);
                        residual(k, weights.k_bias);
                        residual(v, weights.v_bias);
                    }
                    rope(q, position, head_dim, qwen_);
                    rope(k, position, head_dim, qwen_); // v 不做 RoPE

                    // 写入当前层当前位置的 K/V（提交前该层可读）
                    cache.write_layer(layer, k, v);

                    // 多头 attention（含当前 token，因为已写入）
                    std::vector<float> attention(d, 0);
                    for (int head = 0; head < config_.heads; ++head)
                    {
                        const int kv_head = head / sharing; // GQA 映射
                        std::vector<float> scores(position + 1);
                        // 对历史 t ∈ [0, position] 算 score = q·k / sqrt(head_dim)
                        for (std::size_t t = 0; t <= position; ++t)
                        {
                            float dot = 0;
                            for (int i = 0; i < head_dim; ++i)
                                // q[head * head_dim + i] 是当前 token 的 Q
                                // cache.key(layer, t, kv_head, i) 是历史 token t 的 K
                                dot += q[head * head_dim + i] * cache.key(layer, t, kv_head, i);
                            scores[t] = dot / std::sqrt(static_cast<float>(head_dim));
                        }
                        // 数值稳定 softmax
                        const float maximum = *std::max_element(scores.begin(), scores.end());
                        float sum = 0;
                        for (auto &score : scores)
                        {
                            score = std::exp(score - maximum);
                            sum += score;
                        }
                        for (auto &score : scores)
                            score /= sum;
                        // 加权求和 value，累加到 attention[head*head_dim + i]
                        for (std::size_t t = 0; t <= position; ++t)
                            for (int i = 0; i < head_dim; ++i)
                                attention[head * head_dim + i] +=
                                    scores[t] * cache.value(layer, t, kv_head, i);
                    }
                    residual(x, matvec(weights.projection, attention, d));

                    // ---- SwiGLU FFN 子层 ----
                    normalized = norm(x, weights.ffn_norm, epsilon);
                    auto gate = matvec(weights.gate, normalized, config_.hidden_dim);
                    auto up = matvec(weights.up, normalized, config_.hidden_dim);
                    for (std::size_t i = 0; i < gate.size(); ++i)
                        gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i]; // silu(gate)*up
                    residual(x, matvec(weights.down, gate, d));
                }

                // 最终 norm + 分类器 → logits
                auto logits = matvec(classifier_.empty() ? embedding_ : classifier_,
                                     norm(x, final_norm_, qwen_ ? 1e-6f : 1e-5f), config_.vocab);
                check_cancel(cancel);
                // 所有层与最终输出都成功，才提交本 token
                cache.commit_token();
                return logits;
            }
            catch (...)
            {
                // 任何失败：放弃整个请求的缓存状态，再重抛
                cache.reset();
                throw;
            }
        }

        // ---------- 块级预填充：权重对多个 token 复用，注意力保持因果掩码 ----------
        Logits Model::prefill(const std::vector<TokenId> &ids, KVCache &cache,
                              std::size_t chunk_size, const Cancel &cancel) const
        {
            validate_cache(cache);
            if (ids.empty() || chunk_size == 0 || cache.valid_len() != 0)
                throw std::invalid_argument(
                    "prefill requires nonempty input, positive chunk and empty cache");
            if (ids.size() > cache.shape().capacity)
                throw std::out_of_range("mini prompt too long");
            for (auto id : ids)
                if (id < 0 || id >= config_.vocab)
                    throw std::out_of_range("mini token out of range");

            const auto dim = static_cast<std::size_t>(config_.dim);
            const auto head_dim = config_.dim / config_.heads;
            const auto kv_dim = static_cast<std::size_t>(head_dim * config_.kv_heads);
            const auto sharing = config_.heads / config_.kv_heads;
            const float epsilon = qwen_ ? 1e-6f : 1e-5f;
            Logits logits;
            try
            {
                for (std::size_t offset = 0; offset < ids.size();)
                {
                    const auto count = std::min(chunk_size, ids.size() - offset);
                    const auto base = cache.valid_len();
                    std::vector<float> x(count * dim);
                    for (std::size_t token = 0; token < count; ++token)
                        std::copy_n(embedding_.begin() + ids[offset + token] * dim, dim,
                                    x.begin() + token * dim);

                    std::vector<std::vector<float>> block_keys, block_values;
                    block_keys.reserve(layers_.size());
                    block_values.reserve(layers_.size());
                    for (std::size_t layer = 0; layer < layers_.size(); ++layer)
                    {
                        check_cancel(cancel);
                        const auto &weights = layers_[layer];
                        auto normalized = norm_block(x, weights.attention_norm, epsilon);
                        auto q = matmul_block(weights.q, normalized, count, dim,
                                              config_.dim, cancel);
                        auto k = matmul_block(weights.k, normalized, count, dim,
                                              static_cast<int>(kv_dim), cancel);
                        auto v = matmul_block(weights.v, normalized, count, dim,
                                              static_cast<int>(kv_dim), cancel);

                        for (std::size_t token = 0; token < count; ++token)
                        {
                            const auto q_offset = token * dim;
                            const auto kv_offset = token * kv_dim;
                            if (qwen_)
                            {
                                for (std::size_t i = 0; i < dim; ++i)
                                    q[q_offset + i] += weights.q_bias[i];
                                for (std::size_t i = 0; i < kv_dim; ++i)
                                {
                                    k[kv_offset + i] += weights.k_bias[i];
                                    v[kv_offset + i] += weights.v_bias[i];
                                }
                            }
                            std::vector<float> q_token(q.begin() + q_offset,
                                                       q.begin() + q_offset + dim);
                            std::vector<float> k_token(k.begin() + kv_offset,
                                                       k.begin() + kv_offset + kv_dim);
                            rope(q_token, base + token, head_dim, qwen_);
                            rope(k_token, base + token, head_dim, qwen_);
                            std::copy(q_token.begin(), q_token.end(), q.begin() + q_offset);
                            std::copy(k_token.begin(), k_token.end(), k.begin() + kv_offset);
                        }

                        std::vector<float> attention(count * dim, 0.0f);
                        for (std::size_t token = 0; token < count; ++token)
                        {
                            check_cancel(cancel);
                            const auto position = base + token;
                            for (int head = 0; head < config_.heads; ++head)
                            {
                                const auto kv_head = head / sharing;
                                std::vector<float> scores(position + 1);
                                for (std::size_t past = 0; past <= position; ++past)
                                {
                                    float dot = 0;
                                    for (int i = 0; i < head_dim; ++i)
                                    {
                                        const float key = past < base
                                                              ? cache.key(layer, past, kv_head, i)
                                                              : k[(past - base) * kv_dim + kv_head * head_dim + i];
                                        dot += q[token * dim + head * head_dim + i] * key;
                                    }
                                    scores[past] = dot / std::sqrt(static_cast<float>(head_dim));
                                }
                                const float maximum =
                                    *std::max_element(scores.begin(), scores.end());
                                float sum = 0;
                                for (auto &score : scores)
                                {
                                    score = std::exp(score - maximum);
                                    sum += score;
                                }
                                for (auto &score : scores)
                                    score /= sum;
                                for (std::size_t past = 0; past <= position; ++past)
                                    for (int i = 0; i < head_dim; ++i)
                                    {
                                        const float value = past < base
                                                                ? cache.value(layer, past, kv_head, i)
                                                                : v[(past - base) * kv_dim + kv_head * head_dim + i];
                                        attention[token * dim + head * head_dim + i] +=
                                            scores[past] * value;
                                    }
                            }
                        }

                        residual(x, matmul_block(weights.projection, attention, count,
                                                 dim, config_.dim, cancel));
                        normalized = norm_block(x, weights.ffn_norm, epsilon);
                        auto gate = matmul_block(weights.gate, normalized, count, dim,
                                                 config_.hidden_dim, cancel);
                        const auto up = matmul_block(weights.up, normalized, count, dim,
                                                     config_.hidden_dim, cancel);
                        for (std::size_t i = 0; i < gate.size(); ++i)
                            gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
                        residual(x, matmul_block(weights.down, gate, count,
                                                 static_cast<std::size_t>(config_.hidden_dim),
                                                 config_.dim, cancel));
                        block_keys.push_back(std::move(k));
                        block_values.push_back(std::move(v));
                    }

                    // 只有整个 prompt 的最后一个位置需要词表投影。
                    if (offset + count == ids.size())
                    {
                        const std::vector<float> last(x.end() - dim, x.end());
                        logits = matvec(classifier_.empty() ? embedding_ : classifier_,
                                        norm(last, final_norm_, epsilon), config_.vocab);
                        check_cancel(cancel);
                    }
                    cache.append_block(block_keys, block_values, count);
                    offset += count;
                }
                return logits;
            }
            catch (...)
            {
                cache.reset();
                throw;
            }
        }

        // ---------- 单步解码：要求已有前缀 ----------
        Logits Model::decode(TokenId token, KVCache &cache, const Cancel &cancel) const
        {
            if (!cache.valid_len())
                throw std::invalid_argument("decode requires prefilled cache");
            return forward_token(token, cache, cancel);
        }

        // ---------- 对照实现：临时缓存 + 完整 prefill ----------
        Logits Model::forward_full(const std::vector<TokenId> &ids) const
        {
            KVCache temporary(cache_shape());
            return prefill(ids, temporary);
        }

        // ---------- 贪心：argmax，空 logits 拒绝 ----------
        TokenId Model::greedy(const Logits &logits)
        {
            if (logits.empty())
                throw std::invalid_argument("empty mini logits");
            return static_cast<TokenId>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
    }
} // namespace llm_rpc::inference
