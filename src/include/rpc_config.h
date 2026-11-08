#ifndef LLM_RPC_RPC_CONFIG_H
#define LLM_RPC_RPC_CONFIG_H
#include <unordered_map>
#include <string>

namespace llm_rpc {
class RpcConfig{
    public:
    void LoadConfigFile(const char *config_file);//加载配置文件
    std::string Load(const std::string &key);//查找key对应的value
    private:
    std::unordered_map<std::string, std::string> config_map;
    void Trim(std::string &read_buf);//去掉字符串前后的空格
};

}  // namespace llm_rpc

#endif
