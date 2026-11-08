#ifndef LLM_RPC_RPC_LOGGER_H
#define LLM_RPC_RPC_LOGGER_H
#include<glog/logging.h>
#include<string>

namespace llm_rpc {
//采用RAII的思想
class RpcLogger
{
public:
      //构造函数，自动初始化glog
      explicit RpcLogger(const char *argv0)
      {
        google::InitGoogleLogging(argv0);
        FLAGS_colorlogtostderr=true;//启用彩色日志
        FLAGS_logtostderr=true;//默认输出标准错误
      }
      ~RpcLogger(){
        google::ShutdownGoogleLogging();
      }
      //提供静态日志方法
      static void Info(const std::string &message)
      {
        LOG(INFO)<<message;
      }
      static void Warning(const std::string &message){
        LOG(WARNING)<<message;
      }
      static void ERROR(const std::string &message){
        LOG(ERROR)<<message;
      }
          static void Fatal(const std::string& message) {
        LOG(FATAL) << message;
    }
//禁用拷贝构造函数和重载赋值函数
private:
    RpcLogger(const RpcLogger&)=delete;
    RpcLogger& operator=(const RpcLogger&)=delete;
};


}  // namespace llm_rpc

#endif