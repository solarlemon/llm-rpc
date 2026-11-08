#ifndef LLM_RPC_RPC_CHANNEL_H
#define LLM_RPC_RPC_CHANNEL_H

#include <google/protobuf/service.h>
#include <unistd.h>
#include "service_discovery.h"

namespace llm_rpc {

class RpcChannel : public google::protobuf::RpcChannel
{
public:

    RpcChannel(bool connectNow = false) {} 
    
    // 【关键修改点】：析构函数也直接实现
    virtual ~RpcChannel() override {}

    void CallMethod(const ::google::protobuf::MethodDescriptor *method,
                    ::google::protobuf::RpcController *controller,
                    const ::google::protobuf::Message *request,
                    ::google::protobuf::Message *response,
                    ::google::protobuf::Closure *done) override;

private:
    // 内部辅助函数声明
    ssize_t recv_exact(int fd, char *buf, size_t size);
};


}  // namespace llm_rpc

#endif