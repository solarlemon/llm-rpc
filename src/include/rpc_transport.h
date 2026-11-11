#pragma once
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <poll.h>

namespace llm_rpc
{
    // 核心安全限制：
    // 最大帧大小（8MB）。大模型推理的 KV Cache 快照可能很大，但也不能无限制，
    // 防止恶意超大包打爆内存（OOM）。
    constexpr std::uint32_t rpc_max_frame = 8 * 1024 * 1024;

    // 最大协议头大小（4KB）。RpcHeader 只包含 service_name 和 method_name，4KB 绝对够用。
    constexpr std::uint32_t rpc_max_header = 4096;

    // 定义截止时间的类型，使用 steady_clock 保证不受系统时间修改（如 NTP 校时）的影响
    using RpcDeadline = std::chrono::steady_clock::time_point;

    // 等待 Socket 变为可读或可写（带绝对超时时间）
    inline bool WaitSocket(int fd, short events, RpcDeadline deadline)
    {
        for (;;)
        {
            // 1. 计算距离截止时间还剩多少时间
            const auto remaining = deadline - std::chrono::steady_clock::now();

            // 2. 如果剩余时间 <= 0，说明已经超时，直接返回 false
            if (remaining <= decltype(remaining)::zero())
                return false;

            // 3. 将剩余时间转换为毫秒
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count();

            // 4. 组装 poll 的监听结构体
            pollfd descriptor{fd, events, 0};

            // 5. 调用系统调用 poll 进行等待
            // 注意：这里传入了 ms + 1，是因为 duration_cast 是向下取整的。
            // 比如剩余 1.5ms，强转成 ms 是 1。加 1 防止过早超时。
            const int result = poll(&descriptor, 1, static_cast<int>(ms + 1));

            // 6. 处理 poll 的返回值
            if (result > 0)
            {
                // 如果返回值 > 0，说明事件就绪。
                // 检查是不是我们要的事件（可读/可写），或者发生了错误/挂断（POLLERR, POLLHUP）
                return (descriptor.revents & (events | POLLERR | POLLHUP)) != 0;
            }

            // 7. 处理错误和中断
            if (result < 0 && errno != EINTR)
            {
                // 如果 poll 失败，且不是因为被信号中断（EINTR），直接返回 false
                return false;
            }

            // 如果 result == 0（超时），或者发生了 EINTR，进入下一次循环重新计算剩余时间
        }
    }
}