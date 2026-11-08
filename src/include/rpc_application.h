#ifndef LLM_RPC_RPC_APPLICATION_H
#define LLM_RPC_RPC_APPLICATION_H
#include "rpc_config.h"
#include "rpc_channel.h" 
#include  "rpc_controller.h"
#include<mutex>

namespace llm_rpc {
//LLM-RPC基础类，负责框架的一些初始化操作
class RpcApplication
{
    public:
    static void Init(int argc,char **argv);
    static RpcApplication & GetInstance();
    static void deleteInstance();
    static RpcConfig& GetConfig();
    private:
    static RpcConfig m_config;
    static RpcApplication * m_application;//全局唯一单例访问对象
    static std::mutex m_mutex;
    RpcApplication(){}
    ~RpcApplication(){}
    RpcApplication(const RpcApplication&)=delete;
    RpcApplication(RpcApplication&&)=delete;
};

}  // namespace llm_rpc

#endif 
