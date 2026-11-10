#include <iostream>
#include <string>
#include "../user.pb.h"      // protoc 生成：UserServiceRpc 基类、LoginRequest/Response
#include "rpc_application.h" // 框架初始化、配置
#include "rpc_provider.h"    // 服务端核心：NotifyService / Run / OnMessage

/*
UserService 原本是一个本地服务，提供了两个本地方法：Login 和 GetFriendLists。
现在通过 RPC 框架，这些方法可以被远程调用。
*/
class UserService : public llm_rpc::example::UserServiceRpc
//                  ↑ 继承 protoc 生成的 RPC 服务基类
//                  这样 UserService 才能被框架当作"可调用的服务"注册进去
{
public:
    // 真正的"登录逻辑"在这里，protobuf / 框架都不认识它
    bool Login(std::string name, std::string pwd)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // 模拟查库 5ms
        return true;                                               // 模拟登录成功
    }

    // Login ：重写基类虚函数（RPC 框架入口）
    // 签名必须与 UserServiceRpc::Login 完全一致
    // 框架通过 CallMethod 的虚函数分派最终调用到这里
    void Login(::google::protobuf::RpcController *controller,
               const ::llm_rpc::example::LoginRequest *request,
               ::llm_rpc::example::LoginResponse *response,
               ::google::protobuf::Closure *done)
    {
        // 1. 从请求消息中取出参数（request 已由框架反序列化好）
        std::string name = request->name();
        std::string pwd = request->pwd();

        // 2. 调用本地Login业务逻辑
        bool login_result = Login(name, pwd);

        // 3. 把结果写进 response（框架稍后会序列化它发回客户端）
        llm_rpc::example::ResultCode *code = response->mutable_result();
        code->set_errcode(0);                // 0 = 业务成功
        code->set_errmsg("");                // 无错误信息
        response->set_success(login_result); // 登录结果

        // 4. 触发回包
        // done 是框架传入的 RpcClosure，Run() 会：
        //   - 序列化 response
        //   - conn->send 发回客户端
        //   - delete response/request
        //   - delete this（闭包自毁）
        done->Run();
    }
};

int main(int argc, char **argv)
{
    // 1. 框架初始化：解析命令行、加载配置文件（ip/port/zk 地址等）
    llm_rpc::RpcApplication::Init(argc, argv);

    // 2. 创建服务端对象（内部持有 muduo EventLoop、TcpServer、线程池等）
    llm_rpc::RpcProvider provider;

    // 3. 注册服务：把 UserService 发布到框架
    //    NotifyService 内部会用 protobuf 反射遍历它所有 method，
    //    建立 service_name → {method_name → MethodDescriptor} 的映射，
    //    存进 service_map，供 OnMessage 路由时查询
    provider.NotifyService(new UserService());
    //                        ↑ 堆上创建，所有权交给 provider，
    //                          ~RpcProvider 时负责 delete

    // 4. 启动服务器：绑定端口、注册 ZooKeeper、进入 muduo 事件循环（阻塞）
    //    收到请求后：OnMessage 解包 → 线程池 → CallMethod → 虚分派 → Login ②
    provider.Run();

    return 0;
}