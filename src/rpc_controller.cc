#include "rpc_controller.h"

namespace llm_rpc {

// 构造函数，初始化控制器状态
RpcController::RpcController() {
    m_failed = false;  // 初始状态为未失败
    m_errText = "";    // 错误信息初始为空
}

// 重置控制器状态，将失败标志和错误信息清空
void RpcController::Reset() {
    m_failed = false;  // 重置失败标志
    m_errText = "";    // 清空错误信息
}

// 判断当前RPC调用是否失败
bool RpcController::Failed() const {
    return m_failed;  // 返回失败标志
}

// 获取错误信息
std::string RpcController::ErrorText() const {
    return m_errText;  // 返回错误信息
}

// 设置RPC调用失败，并记录失败原因
void RpcController::SetFailed(const std::string &reason) {
    m_failed = true;   // 设置失败标志
    m_errText = reason; // 记录失败原因
}

// 以下功能未实现，是RPC服务端提供的取消功能
// 开始取消RPC调用（未实现）
void RpcController::StartCancel() {
    // 目前为空，未实现具体功能
}

// 判断RPC调用是否被取消（未实现）
bool RpcController::IsCanceled() const {
    return false;  // 默认返回false，表示未被取消
}

// 注册取消回调函数（未实现）
void RpcController::NotifyOnCancel(google::protobuf::Closure* callback) {
    // 目前为空，未实现具体功能
}
}  // namespace llm_rpc
