#include "rpc_channel.h"
#include "rpc_header.pb.h"
#include "rpc_application.h"
#include "rpc_controller.h"
#include <errno.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include "rpc_logger.h"
#include "rpc_connection_pool.h"

namespace llm_rpc
{

    // 全局请求 ID 生成器（原子自增），供一致性哈希路由使用
    static std::atomic<uint64_t> g_req_id{0};

    // 精确读满 size 字节的工具函数（处理 TCP 粘包/半包）
    ssize_t RpcChannel::recv_exact(int fd, char *buf, size_t size)
    {
        size_t total_read = 0;
        while (total_read < size)
        {
            ssize_t ret = recv(fd, buf + total_read, size - total_read, 0);
            if (ret == 0)
                return 0; // 对端关闭连接
            if (ret == -1)
            {
                if (errno == EINTR)
                    continue; // 被信号打断，重试
                return -1;    // 真错误
            }
            total_read += ret; // 累积已读字节
        }
        return total_read;
    }

    void RpcChannel::CallMethod(
        const ::google::protobuf::MethodDescriptor *method, // 方法描述符（含 service/method 名）
        ::google::protobuf::RpcController *controller,      // 状态控制器（成功/失败/错误文本）
        const ::google::protobuf::Message *request,         // 请求消息（已由 Stub 填好）
        ::google::protobuf::Message *response,              // 响应消息（本函数填充）
        ::google::protobuf::Closure *done)                  // 异步回调（此处未用，同步实现）
    {
        // 从描述符取出服务名和方法名（用于路由和服务端分发）
        std::string service_name = method->service()->name(); // "UserServiceRpc"
        std::string method_name = method->name();             // "Login"

        // 1. 服务发现初始化（只执行一次）
        // 静态 once_flag + call_once：多线程环境下保证 Init 只执行一次
        static std::once_flag init_flag; // C++11 提供的线程安全单次执行机制
        std::call_once(init_flag, []()
                       { ServiceDiscovery::GetInstance().Init(); });

        // 2. 生成请求 ID，作为路由 key
        uint64_t req_id = g_req_id.fetch_add(1, std::memory_order_relaxed); // 全局自增 ID
        std::string route_key = std::to_string(req_id);                     // 配合一致性哈希，把不同请求均匀打到不同服务节点，实现负载均衡

        // 3. 服务发现：通过一致性哈希路由，找到目标服务节点的 ip:port
        std::string ip_port = ServiceDiscovery::GetInstance().GetTargetNode(service_name, route_key);
        if (ip_port.empty())
        {
            controller->SetFailed("Hash ring returned empty node!");
            return;
        }
        // 解析 ip:port
        size_t pos = ip_port.find(":");
        std::string target_ip = ip_port.substr(0, pos);
        uint16_t target_port = atoi(ip_port.substr(pos + 1).c_str());

        // 4. 长连接复用：从池中取一条已建好的 TCP 连接，避免每次握手
        int clientfd = RpcConnectionPool::GetInstance().BorrowConnection(target_ip, target_port);
        if (clientfd == -1)
        {
            controller->SetFailed("Borrow connection from pool failed!");
            return;
        }

        // 5. 构造 RPC 请求包（自定义协议 + protobuf 序列化）
        bool is_bad = false; // 标记连接是否已损坏（决定归还时是否销毁）
        std::string args_str;

        if (!request->SerializeToString(&args_str))
        {
            controller->SetFailed("Serialize request fail");
            is_bad = true; // 没发出去但连接仍可用，保守起见也可 false
        }
        else
        {
            // 构造 RpcHeader（自定义协议头，protobuf 定义）
            llm_rpc::protocol::RpcHeader rpc_header;
            rpc_header.set_service_name(service_name);
            rpc_header.set_method_name(method_name);
            rpc_header.set_args_size(args_str.size());

            std::string rpc_header_str;
            rpc_header.SerializeToString(&rpc_header_str);

            uint32_t header_size = rpc_header_str.size();
            uint32_t total_len = 4 + header_size + args_str.size(); // 4 = header_size 字段本身

            // 转网络字节序（大端），跨平台正确
            uint32_t net_total_len = htonl(total_len);
            uint32_t net_header_len = htonl(header_size);

            // 拼包：[total_len(4)][header_len(4)][header][args]
            std::string send_rpc_str;
            send_rpc_str.reserve(4 + 4 + header_size + args_str.size()); // 预分配，避免多次扩容
            send_rpc_str.append((char *)&net_total_len, 4);
            send_rpc_str.append((char *)&net_header_len, 4);
            send_rpc_str.append(rpc_header_str);
            send_rpc_str.append(args_str);

            // 6. socket 发送 + 接收 + 反序列化
            // MSG_NOSIGNAL：对端关闭时返回 EPIPE 而非触发 SIGPIPE（防止进程被杀）
            if (send(clientfd, send_rpc_str.c_str(), send_rpc_str.size(), MSG_NOSIGNAL) == -1)
            {
                // 默认情况下，向已关闭的 socket send 会触发 SIGPIPE 信号，默认行为是终止进程。
                // MSG_NOSIGNAL 让它改为返回 -1（errno = EPIPE），程序可以优雅处理。
                // 高并发 RPC 场景下对端随时可能断连，不加这个标志会导致客户端崩溃。
                controller->SetFailed("Send rpc request error!");
                is_bad = true;
            }
            else
            {
                uint32_t response_len = 0;

                // 先读 4 字节响应长度
                if (recv_exact(clientfd, (char *)&response_len, 4) != 4)
                {
                    controller->SetFailed("Recv response header error!");
                    is_bad = true;
                }
                else
                {
                    response_len = ntohl(response_len); // 转回主机序

                    // 按长度读响应体
                    std::vector<char> recv_buf(response_len);
                    if (recv_exact(clientfd, recv_buf.data(), response_len) != (ssize_t)response_len)
                    {
                        controller->SetFailed("Recv response body error!");
                        is_bad = true;
                    }
                    else
                    {
                        // 反序列化到 response 对象
                        if (!response->ParseFromArray(recv_buf.data(), response_len))
                        {
                            controller->SetFailed("Parse response error!");
                            is_bad = true;
                        }
                    }
                }
            }
        }

        // 7. 无论成功失败都归还；is_bad=true 的连接池会直接关闭，不放入池
        RpcConnectionPool::GetInstance().ReturnConnection(target_ip, target_port, clientfd, is_bad);
    }
} // namespace llm_rpc
