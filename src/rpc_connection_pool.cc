#include "rpc_connection_pool.h"
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include "rpc_logger.h"

namespace llm_rpc
{

    // 假设每个 IP:Port 最多建立 1024 个长连接（必须是 2 的幂）
    static const int MAX_CONN_SIZE = 1024;

    RpcConnectionPool &RpcConnectionPool::GetInstance()
    {
        static RpcConnectionPool instance; // C++11 保证线程安全的懒初始化
        return instance;
    }

    RpcConnectionPool::RpcConnectionPool() : m_max_conn_per_node(MAX_CONN_SIZE) {}

    RpcConnectionPool::~RpcConnectionPool()
    {
        std::lock_guard<std::mutex> lock(m_global_mtx);
        for (auto &pair : m_pools)
        {
            int fd;
            // 把队列里所有空闲 fd 都 pop 出来 close
            while (pair.second->free_fds->pop(fd))
            {
                close(fd);
            }
            delete pair.second; // 释放 bucket（其 unique_ptr 会析构队列）
        }
    }

    // 【新增核心功能】：预热连接
    void RpcConnectionPool::WarmUp(const std::string &ip, uint16_t port, int count)
    {
        std::string key = ip + ":" + std::to_string(port);
        ConnectionBucket *bucket = nullptr;

        // 1. 确保 Bucket 存在
        {
            std::lock_guard<std::mutex> lock(m_global_mtx);
            auto it = m_pools.find(key);
            if (it == m_pools.end())
            {
                bucket = new ConnectionBucket(MAX_CONN_SIZE);
                m_pools[key] = bucket;
            }
            else
            {
                bucket = it->second;
            }
        }

        int success_count = 0;
        // 2. 循环建立连接
        for (int i = 0; i < count; ++i)
        {
            // 检查是否超过最大连接限制
            if (bucket->active_count.load() >= m_max_conn_per_node)
                break;

            int fd = socket(AF_INET, SOCK_STREAM, 0);
            if (fd == -1)
                continue;

            // 设置非阻塞/禁用Nagle
            int opt = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

            struct sockaddr_in addr;
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            addr.sin_addr.s_addr = inet_addr(ip.c_str());

            // 执行握手
            if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0)
            {
                // 成功后，原子增加计数并推入无锁队列
                bucket->active_count.fetch_add(1);
                if (!bucket->free_fds->push(fd))
                {
                    // 队列若满（理论上由 active_count 控制不会发生），则回滚
                    close(fd);
                    bucket->active_count.fetch_sub(1);
                }
                else
                {
                    success_count++;
                }
            }
            else
            {
                close(fd);
            }
        }
        LOG(INFO) << "[ConnectPool] WarmUp complete: " << success_count
                  << " connections ready for " << key;
    }

    int RpcConnectionPool::BorrowConnection(const std::string &ip, uint16_t port)
    {
        std::string key = ip + ":" + std::to_string(port);

        // ---- 1. 找 bucket（短临界区）----
        ConnectionBucket *bucket = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_global_mtx);
            auto it = m_pools.find(key);
            if (it == m_pools.end())
            {
                bucket = new ConnectionBucket(MAX_CONN_SIZE); // 首次访问，懒创建
                m_pools[key] = bucket;
            }
            else
            {
                bucket = it->second;
            }
        }

        int fd = -1;

        // ---- 2. 优先从无锁队列取（预热过的连接直接命中，最快路径）----
        if (bucket->free_fds->pop(fd))
        {
            return fd; // ★ 高频路径：零锁、无系统调用
        }

        // ---- 3. 队列为空才动态建连（受 active_count 上限约束）----
        if (bucket->active_count.load(std::memory_order_relaxed) < m_max_conn_per_node)
        {
            // fetch_add 返回旧值：旧值 < 上限 才算抢到名额
            if (bucket->active_count.fetch_add(1) < m_max_conn_per_node)
            {
                fd = socket(AF_INET, SOCK_STREAM, 0);
                int opt = 1;
                setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

                struct sockaddr_in addr;
                addr.sin_family = AF_INET;
                addr.sin_port = htons(port);
                addr.sin_addr.s_addr = inet_addr(ip.c_str());

                if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0)
                {
                    return fd; // 新建成功，直接返回（不回队列）
                }
                else
                {
                    close(fd);
                    bucket->active_count.fetch_sub(1); // 建连失败，回滚名额
                    return -1;
                }
            }
            else
            {
                bucket->active_count.fetch_sub(1); // 抢名额失败，回滚
            }
        }

        return -1; // 队列空 + 已达上限 → 借不到
    }

    void RpcConnectionPool::ReturnConnection(const std::string &ip, uint16_t port, int fd, bool is_bad)
    {
        std::string key = ip + ":" + std::to_string(port);

        // 找 bucket（若不存在说明异常，直接关掉）
        ConnectionBucket *bucket = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_global_mtx);
            auto it = m_pools.find(key);
            if (it != m_pools.end())
                bucket = it->second;
        }

        if (!bucket)
        {
            close(fd);
            return;
        }

        if (is_bad)
        {
            // 坏连接：协议状态可能已错乱（半包、脏数据），不能复用
            close(fd);
            bucket->active_count.fetch_sub(1); // 名额释放
        }
        else
        {
            // 好连接：放回队列复用
            if (!bucket->free_fds->push(fd))
            {
                // 队列满（异常）→ 关掉并释放名额
                close(fd);
                bucket->active_count.fetch_sub(1);
            }
            // 注意：正常入队时 active_count 保持不变（连接还在池里，只是从"用"变"闲"）
        }
    }
} // namespace llm_rpc
