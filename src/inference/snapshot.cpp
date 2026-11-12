#include "snapshot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace llm_rpc
{
    namespace inference
    {
        namespace
        {
            // 魔数与版本常量
            constexpr std::uint32_t version = 1;
            constexpr std::uint32_t fp32 = 1;                           // 数据类型：1代表 32位浮点
            constexpr std::uint32_t layer_major = 1;                    // 内存布局：1代表以层为主序
            constexpr std::uint64_t header_bytes = 64;                  // 头部固定 64 字节
            constexpr std::uint64_t max_cache_bytes = 64 * 1024 * 1024; // 最大允许 64MB Cache

            // 向输出流写入指定字节宽度的整数（小端序）
            void put(std::ostream &out, std::uint64_t value, unsigned width = 4)
            {
                for (unsigned i = 0; i < width; ++i)
                    out.put(static_cast<char>((value >> (i * 8)) & 0xff));
            }

            // 从输入流读取指定字节宽度的整数
            std::uint64_t get(std::istream &in, unsigned width = 4)
            {
                std::uint64_t value = 0;
                for (unsigned i = 0; i < width; ++i)
                {
                    const int byte = in.get();
                    if (byte == std::char_traits<char>::eof())
                        throw std::runtime_error("truncated KV snapshot");
                    value |= static_cast<std::uint64_t>(byte) << (i * 8);
                }
                return value;
            }

            // 计算实际有效载荷的字节数（K 和 V 各一份，float占4字节，共8字节）
            std::uint64_t payload_size(const KVShape &shape, std::size_t positions)
            {
                return std::uint64_t{8} * shape.layers * positions * shape.kv_heads * shape.head_dim;
            }

            // 校验 Token 是否合法
            void validate_token(std::uint64_t token, const Model &model)
            {
                if (token >= static_cast<std::uint64_t>(model.config().vocab))
                    throw std::runtime_error("invalid snapshot first token");
            }
        }

        // 核心写入逻辑：把内存中的 KVCache 拍平写入流
        void KVSnapshot::write(std::ostream &out, const Model &model, const KVCache &cache,
                               TokenId first_token, std::uint32_t request_id)
        {
            const auto shape = model.cache_shape();
            const auto actual = cache.shape();
            const auto positions = cache.valid_len();

            // 防御性校验
            validate_token(static_cast<std::uint64_t>(first_token), model);
            if (actual.layers != shape.layers || actual.capacity != shape.capacity ||
                actual.kv_heads != shape.kv_heads || actual.head_dim != shape.head_dim ||
                positions == 0 || std::any_of(cache.written_.begin(), cache.written_.end(), [](unsigned char written)
                                              { return written != 0; }))
                throw std::runtime_error("snapshot requires a committed cache matching the model");
            if (cache.allocated_bytes() > max_cache_bytes)
                throw std::runtime_error("snapshot cache exceeds byte budget");

            // 写入固定 64 字节的 Header
            out.write("LLKV", 4); // 魔数：LLaMA KV
            for (const auto field : {version, fp32, layer_major,
                                     static_cast<std::uint32_t>(shape.layers), static_cast<std::uint32_t>(shape.capacity),
                                     static_cast<std::uint32_t>(shape.kv_heads), static_cast<std::uint32_t>(shape.head_dim),
                                     static_cast<std::uint32_t>(positions), static_cast<std::uint32_t>(positions),
                                     static_cast<std::uint32_t>(first_token), request_id})
                put(out, field);
            put(out, model.fingerprint(), 8); // 写入模型指纹（防止模型不匹配）
            put(out, cache.valid_bytes(), 8); // 写入有效数据长度

            // 写入实际的浮点数据（只写入有效位置，不写空余容量）
            const auto elements = positions * cache.layer_width_;
            for (std::size_t layer = 0; layer < shape.layers; ++layer)
                for (const auto *buffer : {&cache.keys_, &cache.values_})
                    for (std::size_t index = 0; index < elements; ++index)
                    {
                        const float value = (*buffer)[layer * cache.layer_stride_ + index];
                        if (!std::isfinite(value)) // 防止 NaN 或 Inf 污染
                            throw std::runtime_error("nonfinite KV value");
                        std::uint32_t bits;
                        std::memcpy(&bits, &value, sizeof(bits)); // 按照 IEEE754 提取浮点位
                        put(out, bits);
                    }
        }

        // 编码为字符串（供 RPC 发送）
        std::string KVSnapshot::encode(const Model &model, const KVCache &cache, TokenId first_token,
                                       std::uint32_t request_id, std::size_t max_bytes)
        {
            if (max_bytes < header_bytes || cache.valid_bytes() > max_bytes - header_bytes)
                throw std::runtime_error("KV snapshot exceeds RPC byte budget");
            std::ostringstream out(std::ios::binary);
            out.exceptions(std::ios::failbit | std::ios::badbit); // 开启流异常，防止写入失败
            write(out, model, cache, first_token, request_id);
            return out.str();
        }

        // 保存到本地文件（供本地 CLI 调试使用）
        void KVSnapshot::save(const std::string &path, const Model &model,
                              const KVCache &cache, TokenId first_token, std::uint32_t request_id)
        {
            // 使用 mkstemp 创建临时文件，防止写到一半崩溃留下脏文件
            std::string temporary = path + ".tmp.XXXXXX";
            const int fd = mkstemp(&temporary[0]);
            if (fd < 0)
                throw std::runtime_error("cannot create snapshot temporary file");
            close(fd);
            try
            {
                std::ofstream out(temporary, std::ios::binary);
                out.exceptions(std::ios::failbit | std::ios::badbit);
                write(out, model, cache, first_token, request_id);
                out.close();
                // 原子重命名：确保外部看到的要么是完整的文件，要么不存在
                if (std::rename(temporary.c_str(), path.c_str()) != 0)
                    throw std::runtime_error("cannot publish KV snapshot");
            }
            catch (...)
            {
                std::remove(temporary.c_str()); // 失败则清理临时文件
                throw;
            }
        }

        // 从文件加载
        KVSnapshot KVSnapshot::load(const std::string &path, const Model &model)
        {
            std::ifstream in(path, std::ios::binary | std::ios::ate);
            return read(in, model);
        }

        // 从 RPC 字符串解码
        KVSnapshot KVSnapshot::decode(const std::string &bytes, const Model &model)
        {
            if (bytes.size() > header_bytes + max_cache_bytes)
                throw std::runtime_error("KV snapshot exceeds byte budget");
            std::istringstream in(bytes, std::ios::binary | std::ios::ate);
            return read(in, model);
        }

        // 核心读取逻辑：从流中还原 KVCache
        KVSnapshot KVSnapshot::read(std::istream &in, const Model &model)
        {
            in.seekg(0, std::ios::end);
            const auto size = in.tellg();
            if (size < static_cast<std::streamoff>(header_bytes) ||
                size > static_cast<std::streamoff>(header_bytes + max_cache_bytes))
                throw std::runtime_error("invalid KV snapshot file size");
            in.seekg(0);

            // 1. 校验 Header
            char magic[4];
            if (!in.read(magic, sizeof(magic)) || std::memcmp(magic, "LLKV", 4) != 0 ||
                get(in) != version || get(in) != fp32 || get(in) != layer_major)
                throw std::runtime_error("unsupported KV snapshot format");

            // 2. 校验维度是否与当前加载的模型匹配
            const auto shape = model.cache_shape();
            if (get(in) != shape.layers || get(in) != shape.capacity ||
                get(in) != shape.kv_heads || get(in) != shape.head_dim)
                throw std::runtime_error("snapshot shape mismatch");

            const auto positions = get(in);
            const auto next_position = get(in);
            const auto first_token = get(in);
            const auto request_id = get(in);

            // 3. 校验模型指纹
            if (get(in, 8) != model.fingerprint())
                throw std::runtime_error("snapshot model fingerprint mismatch");
            validate_token(first_token, model);
            if (positions == 0 || positions > shape.capacity || next_position != positions)
                throw std::runtime_error("invalid snapshot token position");

            // 4. 校验 Payload 大小是否与预期严格一致
            const auto bytes = get(in, 8);
            if (payload_size(shape, shape.capacity) > max_cache_bytes ||
                bytes != payload_size(shape, positions) ||
                static_cast<std::uint64_t>(size) != header_bytes + bytes)
                throw std::runtime_error("invalid snapshot payload size");

            // 5. 分配内存并读入浮点数
            auto cache = std::make_unique<KVCache>(shape);
            const auto elements = positions * cache->layer_width_;
            for (std::size_t layer = 0; layer < shape.layers; ++layer)
                for (auto *buffer : {&cache->keys_, &cache->values_})
                    for (std::size_t index = 0; index < elements; ++index)
                    {
                        const auto bits = static_cast<std::uint32_t>(get(in));
                        float value;
                        std::memcpy(&value, &bits, sizeof(value));
                        if (!std::isfinite(value))
                            throw std::runtime_error("nonfinite KV value");
                        (*buffer)[layer * cache->layer_stride_ + index] = value;
                    }
            if (in.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("unexpected KV snapshot trailing data");

            // 6. 恢复缓存状态并返回
            cache->valid_len_ = positions;
            return {static_cast<std::uint32_t>(request_id), static_cast<TokenId>(first_token), std::move(cache)};
        }
    }
}