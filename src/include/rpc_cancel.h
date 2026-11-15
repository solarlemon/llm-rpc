#ifndef LLM_RPC_RPC_CANCEL_H
#define LLM_RPC_RPC_CANCEL_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

namespace llm_rpc
{
    // 一次 RPC 调用的取消票据：IO 线程置位，推理线程轮询。
    // 只承载「客户端已断开」这一个信号，超时仍由调用侧的 deadline 负责。
    struct CancelToken
    {
        // 获取 steady_clock 计数的当前时刻（用于断连时刻记录）
        static long long Now() noexcept
        {
            return std::chrono::steady_clock::now().time_since_epoch().count();
        }

        void CancelByDisconnect() noexcept
        {
            // 先写时刻，再发布标志；读者看到标志后能读到时刻
            disconnected_at.store(Now(), std::memory_order_relaxed);
            disconnected.store(true, std::memory_order_release);
        }

        bool Cancelled() const noexcept
        {
            return disconnected.load(std::memory_order_acquire);
        }

        // 断连时刻（steady_clock 计数；0 表示未断连）
        long long DisconnectedAt() const noexcept
        {
            return disconnected_at.load(std::memory_order_relaxed);
        }

        std::atomic<bool> disconnected{false};
        std::atomic<long long> disconnected_at{0};
    };

    // 一条 TCP 连接上所有在飞请求的登记表。
    // 连接经由连接池复用，一条连接上可能同时有多个请求，
    // 因此取消粒度是「该连接上的全部在飞请求」。
    //
    // 用 weak_ptr 登记：请求结束后自动失效，既不需要显式注销
    // （消除「已完成请求被误杀」与悬垂指针），也不会误杀复用了
    // 同一条连接的下一个请求。
    struct ConnectionState
    {
        // 请求发起时，把票登记在这个连接的名册上
        void Register(const std::shared_ptr<CancelToken> &token)
        {
            std::lock_guard<std::mutex> lock(mutex);

            // 1. 顺手清理死票（防止 vector 无限膨胀）
            inflight.erase(std::remove_if(inflight.begin(), inflight.end(),
                                          [](const std::weak_ptr<CancelToken> &weak)
                                          { return weak.expired(); }),
                           inflight.end());

            // 2. 兜底防御：登记时如果发现连接已经断了，直接当场作废
            if (closed.load(std::memory_order_relaxed))
            {
                token->CancelByDisconnect();
                return;
            }

            // 3. 登记入册（用 weak_ptr 防止延长票据生命周期导致内存泄漏）
            inflight.push_back(token);
        }

        // 连接断开时，由 IO 线程调用：团灭所有在飞请求
        void CancelAll()
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto &weak : inflight)
                if (auto token = weak.lock()) // 尝试提升为 shared_ptr
                    token->CancelByDisconnect();
            inflight.clear();
        }

        std::atomic<bool> closed{false};
        std::mutex mutex;
        std::vector<std::weak_ptr<CancelToken>> inflight;
    };
} // namespace llm_rpc

#endif
