#include "rpc_provider.h"
#include "rpc_application.h" // 全局配置
#include "rpc_header.pb.h"   // RpcHeader 协议头
#include "rpc_logger.h"
#include <iostream>

namespace llm_rpc
{

    // ============================================================
    // RpcClosure：一次性回调闭包
    // ------------------------------------------------------------
    // protobuf 的 Closure 是抽象基类，只有一个纯虚 Run()。
    // 框架用它作为 "响应发送" 的延迟回调：
    //   - 业务处理完后调用 done->Run() 触发回包
    //   - Run 内部执行 cb_() 后 delete this（自销毁）
    // 使用 std::function 保存任意可调用对象，比手写子类灵活。
    // ============================================================
    class RpcClosure : public google::protobuf::Closure
    {
    public:
        explicit RpcClosure(std::function<void()> cb) : cb_(std::move(cb)) {}
        void Run() override
        {
            cb_();       // 执行实际回调逻辑（发响应）
            delete this; // 自销毁：Closure 生命周期 = 一次 Run 调用
        }

    private:
        std::function<void()> cb_;
    };

    // ============================================================
    // NotifyService：注册一个服务到路由表
    // ------------------------------------------------------------
    // 利用 protobuf 的 ServiceDescriptor 反射：
    //   - 拿到 service 名字（"UserServiceRpc"）
    //   - 遍历它所有 method，建立 method_name → MethodDescriptor（描述一个 RPC 方法的元数据对象） 映射
    // 之后 OnMessage 收到请求时，靠这个 map 找到目标方法。
    // ============================================================
    void RpcProvider::NotifyService(google::protobuf::Service *service)
    {
        ServiceInfo service_info;
        const google::protobuf::ServiceDescriptor *psd = service->GetDescriptor();
        std::string service_name = psd->name();
        int method_count = psd->method_count();

        std::cout << "service_name=" << service_name << std::endl;

        for (int i = 0; i < method_count; ++i)
        {
            const google::protobuf::MethodDescriptor *pmd = psd->method(i);
            std::string method_name = pmd->name();
            std::cout << "method_name=" << method_name << std::endl;
            // 记录 "Login" → MethodDescriptor，供之后按名字查
            service_info.method_map.emplace(method_name, pmd);
        }
        service_info.service = service; // 保存服务对象（基类指针，支持多态）
        service_map.emplace(service_name, service_info);
    }

    // ============================================================
    // Run：启动服务端
    // ------------------------------------------------------------
    // 步骤：读配置 → 建 TcpServer → 设回调 → 起线程池 → 注册 ZK → loop
    // ============================================================
    void RpcProvider::Run()
    {
        // 1. 从配置文件读 IP / Port
        std::string ip = RpcApplication::GetInstance().GetConfig().Load("rpcserverip");
        int port = atoi(RpcApplication::GetInstance().GetConfig().Load("rpcserverport").c_str());

        // 2. 创建 muduo TcpServer（Reactor 模型）
        muduo::net::InetAddress address(ip, port);
        std::shared_ptr<muduo::net::TcpServer> server =
            std::make_shared<muduo::net::TcpServer>(&event_loop, address, "RpcProvider");

        // 3. 设置连接/消息回调（用 bind 绑定 this）
        server->setConnectionCallback(
            std::bind(&RpcProvider::OnConnection, this, std::placeholders::_1));
        server->setMessageCallback(
            std::bind(&RpcProvider::OnMessage, this,
                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

        // 4. IO 线程数：4 个负责收发字节流
        server->setThreadNum(4);

        // 5. 业务线程池：100 线程执行业务逻辑
        //    与 IO 线程分离，避免慢业务阻塞网络收发
        m_thread_pool.start(100);

        // 6. 连接 ZooKeeper，注册服务供客户端发现
        ZkClient zkclient;
        zkclient.Start();

        for (auto &sp : service_map)
        {
            // 父节点：/UserServiceRpc
            std::string service_path = "/" + sp.first;
            zkclient.Create(service_path.c_str(), nullptr, 0);

            // 子节点（临时节点）：/UserServiceRpc/127.0.0.1:8000
            // ZOO_EPHEMERAL：进程退出即自动删除 → 客户端立即知道节点下线
            std::string ip_port = ip + ":" + std::to_string(port);
            std::string instance_path = service_path + "/" + ip_port;
            zkclient.Create(instance_path.c_str(), ip_port.c_str(),
                            ip_port.length(), ZOO_EPHEMERAL);
        }

        std::cout << "RpcProvider start service at ip:" << ip
                  << " port:" << port << std::endl;

        // 7. 启动服务器 + 进入事件循环（阻塞）
        server->start();
        event_loop.loop();
    }

    // ============================================================
    // OnConnection：连接建立/断开回调
    // ------------------------------------------------------------
    // 简单处理：断开时 shutdown。连接建立无需额外操作。
    // ============================================================
    void RpcProvider::OnConnection(const muduo::net::TcpConnectionPtr &conn)
    {
        if (!conn->connected())
        {
            conn->shutdown(); // 对端已断开，清理
        }
    }

    // ============================================================
    // OnMessage：收到数据 → 解包 → 反序列化 → 丢线程池执行
    // ------------------------------------------------------------
    // 协议格式（与客户端 RpcChannel 对称）：
    //   [total_len(4)][header_len(4)][RpcHeader][args]
    //
    // while 循环：一次可能读到多个完整请求（流水线/粘包），全部处理
    // ============================================================
    void RpcProvider::OnMessage(const muduo::net::TcpConnectionPtr &conn,
                                muduo::net::Buffer *buffer,
                                muduo::Timestamp receive_time)
    {
        while (buffer->readableBytes() >= 4)
        {
            // ---- 步骤 1：读 4 字节 total_len（网络序 → 主机序）----
            uint32_t total_len = 0;
            std::memcpy(&total_len, buffer->peek(), 4);
            total_len = ntohl(total_len);

            // 半包保护：还没收满一整个请求，等下次数据到来
            // readableBytes 必须 >= 4(自身) + total_len
            if (buffer->readableBytes() < 4 + total_len)
            {
                break;
            }

            buffer->retrieve(4); // 消耗掉 total_len 字段

            // ---- 步骤 2：读 4 字节 header_len ----
            uint32_t header_len = 0;
            const char *data_ptr = buffer->peek();
            std::memcpy(&header_len, data_ptr, 4);
            header_len = ntohl(header_len);
            buffer->retrieve(4); // 消耗掉 header_len 字段

            // ---- 步骤 3：切出 RpcHeader 并反序列化 ----
            std::string rpc_header_str(buffer->peek(), header_len);
            llm_rpc::protocol::RpcHeader rpc_header;
            buffer->retrieve(header_len);

            // ---- 步骤 4：切出 args（用 total_len - 4 - header_len 计算）----
            // 注意：这里没用 RpcHeader.args_size，而是从 total_len 反推
            uint32_t args_size = total_len - 4 - header_len;
            std::string args_str(buffer->peek(), args_size);
            buffer->retrieve(args_size);

            // ---- 步骤 5：解析协议头 ----
            if (!rpc_header.ParseFromString(rpc_header_str))
            {
                std::cout << "header parse error" << std::endl;
                return; // 协议头坏 → 无法继续解析，退出（可考虑关连接）
            }

            std::string service_name = rpc_header.service_name();
            std::string method_name = rpc_header.method_name();

            // ---- 步骤 6：查 service_map 找服务 ----
            auto it = service_map.find(service_name);
            if (it == service_map.end())
            {
                std::cout << service_name << " is not exist!" << std::endl;
                return;
            }
            // 再查 method_map 找方法描述符
            auto mit = it->second.method_map.find(method_name);
            if (mit == it->second.method_map.end())
            {
                std::cout << service_name << "." << method_name << " is not exist!" << std::endl;
                return;
            }

            google::protobuf::Service *service = it->second.service;
            const google::protobuf::MethodDescriptor *method = mit->second;

            // ---- 步骤 7：按 method 原型创建 request 并解析 ----
            // GetRequestPrototype(method).New()：返回该方法的请求消息实例
            google::protobuf::Message *request = service->GetRequestPrototype(method).New();
            if (!request->ParseFromString(args_str))
            {
                delete request;
                return;
            }

            // 按 response 原型创建响应对象（业务侧填充）
            google::protobuf::Message *response = service->GetResponsePrototype(method).New();

            // ---- 步骤 8：创建 done 闭包，业务处理完后触发回包 ----
            // 业务代码末尾 done->Run();  所以最后会调用 SendRpcResponse(conn, response, request) 写回客户端
            // 捕获 conn / response / request，等 CallMethod 完成后发响应并释放资源
            google::protobuf::Closure *done = new RpcClosure(
                [this, conn, response, request]()
                {
                    this->SendRpcResponse(conn, response, request); // 写回客户端
                });

            // ---- 步骤 9：丢到线程池执行业务逻辑 ----
            // 注意：request/response/done 都是堆对象，由闭包负责回收
            m_thread_pool.run([service, method, request, response, done]()
                              { service->CallMethod(method, nullptr, request, response, done); });

            // // 空跑框架测试：不用线程池，直接在 IO 线程处理（会阻塞收发）
            // service->CallMethod(method, nullptr, request, response, done);
        }
    }

    // ============================================================
    // SendRpcResponse：序列化响应并写回客户端（由 done->Run() 触发）
    // ------------------------------------------------------------
    // 响应格式：[response_len(4)][response_body]
    // ============================================================
    void RpcProvider::SendRpcResponse(const muduo::net::TcpConnectionPtr &conn,
                                      google::protobuf::Message *response,
                                      google::protobuf::Message *request)
    {
        std::string response_str;
        if (response->SerializeToString(&response_str))
        {
            uint32_t len = response_str.size();
            uint32_t net_len = htonl(len);

            // 拼包：长度头 + 响应体
            std::string send_buf;
            send_buf.resize(4 + len);
            std::memcpy(&send_buf[0], &net_len, 4);
            std::memcpy(&send_buf[4], response_str.data(), len);

            // muduo 的 conn->send 是线程安全的：
            // 业务线程调用 send 时，muduo 会把数据转回对应 IO 线程发送，
            // 因此无需加锁。
            conn->send(send_buf);
        }
        else
        {
            LOG(ERROR) << "serialize response error!";
        }

        // 释放本次请求的堆对象（谁 New 谁 Delete）
        delete response;
        delete request;
    }

    // ============================================================
    // 析构：释放服务对象、退出事件循环
    // ============================================================
    RpcProvider::~RpcProvider()
    {
        std::cout << "~RpcProvider()" << std::endl;
        for (auto &sp : service_map)
        {
            if (sp.second.service != nullptr)
            {
                delete sp.second.service; // 释放 NotifyService(new UserService())
                sp.second.service = nullptr;
            }
        }
        event_loop.quit();
    }
} // namespace llm_rpc