#include "rpc_connection_pool.h"
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
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

    // 【新增核心功能】：预热连接（暂时只在压测登录时使用）
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

    int RpcConnectionPool::BorrowConnection(const std::string &ip, uint16_t port, RpcDeadline deadline)
    {
        if (std::chrono::steady_clock::now() >= deadline)
            return -1;
        std::string key = ip + ":" + std::to_string(port);

        // ---- 1. 找 bucket（短临界区）----
        ConnectionBucket *bucket = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_global_mtx);
            // map中查找是否已有该地址的 bucket，
            auto it = m_pools.find(key); // find() 函数会返回一个迭代器（iterator，类似于指针），变量名为 it
            // 如果找到了：it 会指向那个具体的键值对。
            // 如果没找到：find() 不会返回空指针（nullptr），而是返回一个特殊的迭代器——m_pools.end()。
            if (it == m_pools.end())
            {
                // 首次访问该地址 IP:Port，懒创建 bucket
                bucket = new ConnectionBucket(MAX_CONN_SIZE);
                // 需要当场创建一个新的连接桶（Bucket），并放入 map 中
                m_pools[key] = bucket;
            }
            else
            {
                // 找到了，直接把这个桶的指针拿出来
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
                fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0); // 创建非阻塞 Socket
                if (fd >= 0)
                {
                    // 设置 TCP_NODELAY 禁用 Nagle 算法，减少延迟
                    int option = 1;
                    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &option, sizeof(option));
                    sockaddr_in address{};
                    address.sin_family = AF_INET;
                    address.sin_port = htons(port);
                    if (inet_pton(AF_INET, ip.c_str(), &address.sin_addr) == 1)
                    {
                        // 发起请求后立刻返回 -1，用 poll 等待 TCP 三次握手完成，全程受 deadline 约束，超时立刻放弃。
                        const int result = connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address));
                        if (result == 0)
                            return fd;
                        if (errno == EINPROGRESS && WaitSocket(fd, POLLOUT, deadline)) // 正在握手，等待 POLLOUT
                        {
                            int error = 0;
                            socklen_t length = sizeof(error);
                            // 非阻塞建连的标准收尾动作。检查底层 socket 是否真的连上了，还是报错了
                            if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length) == 0 && error == 0)
                                return fd;
                        }
                    }
                    close(fd);
                }
                bucket->active_count.fetch_sub(1);
                return -1;
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
            // 直接关闭文件描述符，内核回收资源
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
