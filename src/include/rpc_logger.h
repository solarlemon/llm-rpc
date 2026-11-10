#ifndef LLM_RPC_RPC_LOGGER_H
#define LLM_RPC_RPC_LOGGER_H

#include <glog/logging.h>
#include <string>

namespace llm_rpc
{
  // RpcLogger：glog 的 RAII 封装
  // 构造时初始化 glog，析构时关闭 glog
  class RpcLogger
  {
  public:
    // 构造函数：初始化 glog
    explicit RpcLogger(const char *argv0) // argv0 用作日志文件名前缀
    {
      google::InitGoogleLogging(argv0); // 全局初始化，必须最先调用
      FLAGS_colorlogtostderr = true;    // 输出到 stderr 时带颜色
      FLAGS_logtostderr = true;         // 默认输出到 stderr（而非文件）
    }

    // 析构函数：关闭 glog，释放全局资源
    ~RpcLogger()
    {
      google::ShutdownGoogleLogging();
    }

    // 静态日志方法：直接转发到 glog 的 LOG 宏
    static void Info(const std::string &message) { LOG(INFO) << message; }
    static void Warning(const std::string &message) { LOG(WARNING) << message; }
    static void ERROR(const std::string &message) { LOG(ERROR) << message; }
    static void Fatal(const std::string &message) { LOG(FATAL) << message; }

  private:
    // 禁用拷贝：RpcLogger 持有全局资源（glog），拷贝无意义且危险
    RpcLogger(const RpcLogger &) = delete;
    RpcLogger &operator=(const RpcLogger &) = delete;
  };

} // namespace llm_rpc

#endif