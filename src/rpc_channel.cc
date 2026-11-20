#include "rpc_channel.h"
#include "rpc_header.pb.h"
#include "rpc_connection_pool.h"
#include "rpc_transport.h"

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sys/socket.h>

namespace llm_rpc
{
    namespace
    {
        // 【底层网络收发】带超时机制的可靠传输
        bool Transfer(int fd, char *data, std::size_t size, bool sending, RpcDeadline deadline)
        {
            // 只要还没发完/收完，就循环尝试，直到超时或出错
            while (size)
            {
                // 1. 每次读写前，先用 poll 检查 socket 是否就绪，并受 deadline 约束
                if (!WaitSocket(fd, sending ? POLLOUT : POLLIN, deadline))
                    return false;

                // 2. 非阻塞发送/接收 (MSG_DONTWAIT)
                const auto count = sending
                                       ? send(fd, data, size, MSG_DONTWAIT | MSG_NOSIGNAL)
                                       : recv(fd, data, size, MSG_DONTWAIT);

                // 3. 处理被信号打断 (EINTR) 或缓冲区满/空 (EAGAIN/EWOULDBLOCK)
                if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
                    continue; // 重试
                if (count <= 0)
                    return false; // 真正的错误或对端关闭

                data += count;
                size -= count;
            }
            return true;
        }
    }

    void RpcChannel::CallMethod(const google::protobuf::MethodDescriptor *method,
                                google::protobuf::RpcController *controller,
                                const google::protobuf::Message *request,
                                google::protobuf::Message *response,
                                google::protobuf::Closure *)
    {
        // 【统一错误处理】Lambda 闭包，任何失败都在此上报 controller
        const auto fail = [&](const char *reason)
        {
            if (controller)
                controller->SetFailed(reason);
        };

        // 【全局 Deadline】计算本次 RPC 调用的绝对超时时间点，只要超时立即返回失败，杜绝线程死锁
        const auto deadline = std::chrono::steady_clock::now() + timeout_;

        //  1：序列化请求
        std::string args;
        if (request->ByteSizeLong() > rpc_max_frame || !request->SerializeToString(&args))
            return fail("RPC request exceeds frame limit or serialization failed");

        //  2：构建协议头
        protocol::RpcHeader header;
        header.set_service_name(method->service()->name());
        header.set_method_name(method->name());
        header.set_args_size(args.size());

        std::string encoded;
        // 严格校验：Header 不能超限，且总包大小不能超限（防 OOM）
        if (!header.SerializeToString(&encoded) || encoded.size() > rpc_max_header ||
            4 + encoded.size() + args.size() > rpc_max_frame)
            return fail("RPC request exceeds frame limit");

        //  3：组包（长度前缀协议）
        // [总长 4B] [Header长 4B] [Header] [Args]
        const auto total = htonl(4 + encoded.size() + args.size());
        const auto head = htonl(encoded.size());
        std::string frame(reinterpret_cast<const char *>(&total), sizeof(total));
        frame.append(reinterpret_cast<const char *>(&head), sizeof(head));
        frame += encoded;
        frame += args;

        //  4：初始化性能追踪
        // 可通过环境变量 LLM_RPC_TRACE=1 开启诊断，打印服务发现、借连接、发送、接收等耗时，单位微秒
        const bool trace = std::getenv("LLM_RPC_TRACE") != nullptr;
        using TraceClock = std::chrono::steady_clock;
        const auto trace_us = [](TraceClock::duration span)
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(span).count();
        };

        //  5：服务发现（找服务端 IP:Port）
        static std::once_flag initialized;
        static std::atomic<std::uint64_t> sequence{0};
        const auto discovery_begin = TraceClock::now();
        std::string address;
        try
        {
            // 保证 ServiceDiscovery 只初始化一次
            std::call_once(initialized, []
                           { ServiceDiscovery::GetInstance().Init(); });
            // 利用全局自增序号做一致性哈希的 key，选出目标节点
            // 因为每次请求的 ID 都不同，它能将流量均匀打散到所有的服务端节点上，实现无锁的随机负载均衡。
            address = ServiceDiscovery::GetInstance().GetTargetNode(
                header.service_name(), std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        }
        catch (const std::exception &)
        {
            return fail("RPC service discovery failed");
        }
        const auto discovery_end = TraceClock::now();

        //  6：解析 IP 和端口
        const auto separator = address.find(':');
        if (separator == std::string::npos)
            return fail("No RPC endpoint");
        const auto ip = address.substr(0, separator);
        const auto port_text = address.substr(separator + 1);
        if (port_text.empty() || port_text.size() > 5 ||
            port_text.find_first_not_of("0123456789") != std::string::npos)
            return fail("Invalid RPC endpoint");
        const auto port = std::stoi(port_text);
        if (port <= 0 || port > 65535)
            return fail("Invalid RPC endpoint");

        //  7：从连接池借连接
        auto &pool = RpcConnectionPool::GetInstance();
        const auto borrow_begin = TraceClock::now();
        const auto fd = pool.BorrowConnection(ip, port, deadline); // 借连接也受 deadline 约束
        const auto borrow_end = TraceClock::now();
        if (fd < 0)
            return fail("RPC connection failed or deadline exceeded");

        //  8：发送与接收
        bool complete = false;
        std::uint32_t size = 0;
        const auto send_begin = TraceClock::now();
        const bool sent = Transfer(fd, &frame[0], frame.size(), true, deadline);
        const auto send_end = TraceClock::now();

        // 先收 4 字节长度，再收消息体
        if (sent && Transfer(fd, reinterpret_cast<char *>(&size), sizeof(size), false, deadline))
        {
            size = ntohl(size);
            if (size <= rpc_max_frame)
            {
                std::vector<char> body(size);
                complete = Transfer(fd, body.data(), body.size(), false, deadline) &&
                           response->ParseFromArray(size ? body.data() : "", size);
            }
        }
        const auto recv_end = TraceClock::now();

        //  9：归还连接
        // 如果 complete = false，连接可能处于半包/脏数据状态，标记为坏连接销毁
        pool.ReturnConnection(ip, port, fd, !complete);

        //  10：打印性能追踪日志
        if (trace)
        {
            std::cerr << "[channel] " << header.service_name()
                      << " discovery=" << trace_us(discovery_end - discovery_begin)
                      << "us borrow=" << trace_us(borrow_end - borrow_begin)
                      << "us send=" << trace_us(send_end - send_begin)
                      << "us recv+parse=" << trace_us(recv_end - send_end)
                      << "us bytes=" << size << "\n";
        }

        if (!complete)
            fail("RPC I/O failed, frame invalid, or deadline exceeded");
    }
}
