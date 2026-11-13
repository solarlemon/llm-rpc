#pragma once
#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace llm_rpc {
class QwenTokenizer {
public:
    explicit QwenTokenizer(const std::string &directory);
    std::vector<std::int32_t> prompt(const std::string &question) const;
    std::string decode(const std::vector<std::int32_t> &ids) const;

private:
    struct Data;
    std::shared_ptr<const Data> data_;
};
}
