#pragma once
#include "model.hpp"
#include <memory>
#include <iosfwd>

namespace llm_rpc
{
    namespace inference
    {
        struct KVSnapshot
        {
            std::uint32_t request_id;
            TokenId first_token;
            std::unique_ptr<KVCache> cache;

            static void save(const std::string &path, const Model &model,
                             const KVCache &cache, TokenId first_token, std::uint32_t request_id);
            static KVSnapshot load(const std::string &path, const Model &model);
            static std::string encode(const Model &model, const KVCache &cache, TokenId first_token,
                                      std::uint32_t request_id, std::size_t max_bytes);
            static KVSnapshot decode(const std::string &bytes, const Model &model);

        private:
            static void write(std::ostream &out, const Model &model, const KVCache &cache,
                              TokenId first_token, std::uint32_t request_id);
            static KVSnapshot read(std::istream &in, const Model &model);
        };
    }
}
