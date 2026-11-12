#ifndef LLM_RPC_RPC_CONNECTION_POOL_H
#define LLM_RPC_RPC_CONNECTION_POOL_H

#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <memory>
#include "rpc_transport.h"
#include "lock_free_queue.h"      // MPMCQueue：多生产者多消费者无锁队列

namespace llm_rpc
{
    // ConnectionBucket：每个目标地址（ip:port）对应一个桶
    // 存放该地址的所有空闲 fd + 连接总数
    struct ConnectionBucket
    {
        // 空闲 fd 队列：多线程并发 push/pop 无锁
        std::unique_ptr<MPMCQueue<int>> free_fds;

        // 该地址已建立的连接总数（空闲 + 正在使用）
        // 用于限制总连接数，防止把服务端打爆
        std::atomic<int> active_count;

        ConnectionBucket(size_t capacity)
            : free_fds(new MPMCQueue<int>(capacity)), active_count(0) {}
    };

    // RpcConnectionPool：连接池单例
    class RpcConnectionPool
    {
    public:
        static RpcConnectionPool &GetInstance();     // 单例（C++11 magic static）

        int  BorrowConnection(const std::string &ip, uint16_t port,
            RpcDeadline deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10));   // 借连接
        void ReturnConnection(const std::string &ip, uint16_t port,
                              int fd, bool is_bad = false);            // 还连接
        void WarmUp(const std::string &ip, uint16_t port, int count);  // 预热

    private:
        RpcConnectionPool();
        ~RpcConnectionPool();                        // 析构：关闭所有 fd，释放 bucket

        int m_max_conn_per_node;                     // 每地址连接上限（1024）
        std::unordered_map<std::string, ConnectionBucket *> m_pools;  // "ip:port" → bucket
        std::mutex m_global_mtx;                     // 只保护 m_pools 的查找/插入
    };

} // namespace llm_rpc

#endif