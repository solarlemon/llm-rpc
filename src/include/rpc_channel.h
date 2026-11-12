#ifndef LLM_RPC_RPC_CHANNEL_H
#define LLM_RPC_RPC_CHANNEL_H

#include <google/protobuf/service.h>
#include <chrono>
#include "service_discovery.h"

namespace llm_rpc {

class RpcChannel : public google::protobuf::RpcChannel
{
public:

    explicit RpcChannel(bool = false,
        std::chrono::milliseconds timeout = std::chrono::seconds(10)) : timeout_(timeout) {}
    
    // 【关键修改点】：析构函数也直接实现
    virtual ~RpcChannel() override {}

    void CallMethod(const ::google::protobuf::MethodDescriptor *method,
                    ::google::protobuf::RpcController *controller,
                    const ::google::protobuf::Message *request,
                    ::google::protobuf::Message *response,
                    ::google::protobuf::Closure *done) override;

private:
    std::chrono::milliseconds timeout_;
};


}  // namespace llm_rpc

#endif