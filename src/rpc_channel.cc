#include "rpc_channel.h"
#include "rpc_header.pb.h"
#include "rpc_connection_pool.h"
#include "rpc_transport.h"

#include <arpa/inet.h>
#include <atomic>
#include <mutex>
#include <sys/socket.h>

namespace llm_rpc
{
    namespace
    {
        bool Transfer(int fd, char *data, std::size_t size, bool sending, RpcDeadline deadline)
        {
            while (size)
            {
                if (!WaitSocket(fd, sending ? POLLOUT : POLLIN, deadline))
                    return false;
                const auto count = sending
                                       ? send(fd, data, size, MSG_DONTWAIT | MSG_NOSIGNAL)
                                       : recv(fd, data, size, MSG_DONTWAIT);
                if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
                    continue;
                if (count <= 0)
                    return false;
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
        const auto fail = [&](const char *reason)
        {
            if (controller)
                controller->SetFailed(reason);
        };
        const auto deadline = std::chrono::steady_clock::now() + timeout_;
        std::string args;
        if (request->ByteSizeLong() > rpc_max_frame || !request->SerializeToString(&args))
            return fail("RPC request exceeds frame limit or serialization failed");
        protocol::RpcHeader header;
        header.set_service_name(method->service()->name());
        header.set_method_name(method->name());
        header.set_args_size(args.size());
        std::string encoded;
        if (!header.SerializeToString(&encoded) || encoded.size() > rpc_max_header ||
            4 + encoded.size() + args.size() > rpc_max_frame)
            return fail("RPC request exceeds frame limit");
        const auto total = htonl(4 + encoded.size() + args.size());
        const auto head = htonl(encoded.size());
        std::string frame(reinterpret_cast<const char *>(&total), sizeof(total));
        frame.append(reinterpret_cast<const char *>(&head), sizeof(head));
        frame += encoded;
        frame += args;

        static std::once_flag initialized;
        static std::atomic<std::uint64_t> sequence{0};
        std::string address;
        try
        {
            std::call_once(initialized, []
                           { ServiceDiscovery::GetInstance().Init(); });
            address = ServiceDiscovery::GetInstance().GetTargetNode(
                header.service_name(), std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
        }
        catch (const std::exception &)
        {
            return fail("RPC service discovery failed");
        }
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
        auto &pool = RpcConnectionPool::GetInstance();
        const auto fd = pool.BorrowConnection(ip, port, deadline);
        if (fd < 0)
            return fail("RPC connection failed or deadline exceeded");

        bool complete = false;
        std::uint32_t size = 0;
        if (Transfer(fd, &frame[0], frame.size(), true, deadline) &&
            Transfer(fd, reinterpret_cast<char *>(&size), sizeof(size), false, deadline))
        {
            size = ntohl(size);
            if (size <= rpc_max_frame)
            {
                std::vector<char> body(size);
                complete = Transfer(fd, body.data(), body.size(), false, deadline) &&
                           response->ParseFromArray(size ? body.data() : "", size);
            }
        }
        pool.ReturnConnection(ip, port, fd, !complete);
        if (!complete)
            fail("RPC I/O failed, frame invalid, or deadline exceeded");
    }
}
