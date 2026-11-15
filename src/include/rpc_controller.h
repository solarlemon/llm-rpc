#ifndef LLM_RPC_RPC_CONTROLLER_H
#define LLM_RPC_RPC_CONTROLLER_H

#include <google/protobuf/service.h>
#include <memory>
#include <string>
#include "rpc_cancel.h"

namespace llm_rpc
{
    // 用于描述RPC调用的控制器
    // 其主要作用是跟踪RPC方法调用的状态、错误信息并提供控制功能(如取消调用)。
    class RpcController : public google::protobuf::RpcController
    {
    public:
        RpcController();
        void Reset() override;
        bool Failed() const override;
        std::string ErrorText() const override;
        void SetFailed(const std::string &reason) override;

        // 取消目前只实现了「客户端断开连接」这一种来源：
        // 服务端由 RpcProvider 在派发前注入 CancelToken，业务层通过 IsCanceled() 轮询。
        void StartCancel() override;
        bool IsCanceled() const override;
        void NotifyOnCancel(google::protobuf::Closure *callback) override;

        // 服务端侧：由 RpcProvider 注入本次请求的取消票据。客户端侧不调用，保持为空。
        void SetCancelToken(const std::shared_ptr<CancelToken> &token) { m_cancel = token; }
        // 只读访问：worker 用它统计「断连到被观察到」的延迟（客户端侧为空）
        const std::shared_ptr<CancelToken> &GetCancelToken() const { return m_cancel; }

    private:
        bool m_failed;                         // RPC方法执行过程中的状态
        std::string m_errText;                 // RPC方法执行过程中的错误信息
        std::shared_ptr<CancelToken> m_cancel; // 服务端取消票据；客户端为空
    };

} // namespace llm_rpc

#endif
