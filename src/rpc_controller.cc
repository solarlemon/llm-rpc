#include "rpc_controller.h"

namespace llm_rpc
{

    // 构造函数，初始化控制器状态
    RpcController::RpcController()
    {
        m_failed = false; // 初始状态为未失败
        m_errText = "";   // 错误信息初始为空
    }

    // 重置控制器状态，将失败标志和错误信息清空
    void RpcController::Reset()
    {
        m_failed = false; // 重置失败标志
        m_errText = "";   // 清空错误信息
        // 注意：不清 m_cancel。服务端每次请求都会新建控制器，
        // 清掉它会让「控制器复用」场景下的取消检测失效。
    }

    // 判断当前RPC调用是否失败
    bool RpcController::Failed() const
    {
        return m_failed; // 返回失败标志
    }

    // 获取错误信息
    std::string RpcController::ErrorText() const
    {
        return m_errText; // 返回错误信息
    }

    // 设置RPC调用失败，并记录失败原因
    void RpcController::SetFailed(const std::string &reason)
    {
        m_failed = true;    // 设置失败标志
        m_errText = reason; // 记录失败原因
    }

    // 客户端主动取消：协议里没有取消帧，暂未实现。
    // 当前唯一的取消来源是「客户端断开连接」，见 IsCanceled()。
    void RpcController::StartCancel()
    {
    }

    // 双端共用：服务端返回「客户端是否已断开」；客户端控制器没有票据，恒为 false。
    bool RpcController::IsCanceled() const
    {
        return m_cancel && m_cancel->Cancelled();
    }

    // 取消回调：当前不需要，保持空实现。
    void RpcController::NotifyOnCancel(google::protobuf::Closure *callback)
    {
    }

} // namespace llm_rpc
