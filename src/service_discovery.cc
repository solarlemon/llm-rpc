#include "service_discovery.h"
#include "rpc_logger.h"

namespace llm_rpc
{

    // 单例：C++11 静态局部变量，线程安全且懒初始化
    ServiceDiscovery &ServiceDiscovery::GetInstance()
    {
        static ServiceDiscovery instance;
        return instance;
    }

    // Init：初始化 ZK 客户端
    // 由 RpcChannel::CallMethod 里的 std::call_once 保证只调用一次
    void ServiceDiscovery::Init()
    {
        m_zkClient.Start(); // 连接 ZK，阻塞直到会话建立
    }

    // GetTargetNode：核心！给定服务名 + 路由 key，返回目标 ip:port
    // ------------------------------------------------------------
    // 快路径：命中缓存 → 读锁 → 哈希环选节点（无网络）
    // 慢路径：未命中 → 去 ZK 拉取 → 写锁 → 建哈希环 → 选节点
    std::string ServiceDiscovery::GetTargetNode(const std::string &service_name,
                                                const std::string &key)
    {
        // ---------- 快路径：读锁检查缓存 ----------
        {
            std::shared_lock<std::shared_timed_mutex> read_lock(m_rw_mtx);
            if (m_chash_map.find(service_name) != m_chash_map.end())
            {
                // 命中：直接在一致性哈希环上按 key 选节点
                return m_chash_map[service_name]->GetTargetNode(key);
            }
        } // 读锁在此释放

        // ---------- 慢路径：缓存未命中，去 ZK 拉 ----------
        // 注意：网络请求很慢，绝不能持锁做网络 IO！
        // 否则会阻塞所有其他线程的读操作
        std::string path = "/" + service_name;
        std::vector<std::string> new_nodes =
            m_zkClient.GetChildren(path.c_str(), WatcherCallback, this);
        //                                              ↑ 注册 watcher，节点变化时回调

        // ---------- 写锁更新缓存（双重检查）----------
        {
            std::unique_lock<std::shared_timed_mutex> write_lock(m_rw_mtx);
            // 双重检查：另一个线程可能在我们拉取期间已经建好了
            if (m_chash_map.find(service_name) == m_chash_map.end())
            {
                m_chash_map[service_name] = std::unique_ptr<ConsistentHash>(new ConsistentHash());
                m_chash_map[service_name]->UpdateNodes(new_nodes);
                m_nodes_cache[service_name] = new_nodes;
            }
            return m_chash_map[service_name]->GetTargetNode(key);
        }
    }

    // WatcherCallback：ZK 子节点变化回调
    // ------------------------------------------------------------
    // 当 /UserServiceRpc 下的实例节点增删时，ZK 回调此函数。
    // 职责：重新拉取子节点 → 更新哈希环。
    // 关键约束：这里是 ZK 客户端线程调用的，不能阻塞太久、不能加锁做网络 IO
    void ServiceDiscovery::WatcherCallback(zhandle_t *zh, int type, int state,
                                           const char *path, void *watcherCtx)
    {
        if (type == ZOO_CHILD_EVENT) // 子节点变化事件
        {
            ServiceDiscovery *sd = static_cast<ServiceDiscovery *>(watcherCtx);
            std::string path_str = path;
            std::string service_name = path_str.substr(1); // "/UserServiceRpc" → "UserServiceRpc"

            // 千万不要在这里加锁！先发起阻塞网络请求获取最新节点
            std::vector<std::string> new_nodes =
                sd->m_zkClient.GetChildren(path_str.c_str(), WatcherCallback, sd);
            //                                              ↑ 重新注册 watcher（一次性，必须重注册）

            // 通过 ZK 拿到数据后，再加写锁更新内存结构
            std::unique_lock<std::shared_timed_mutex> lock(sd->m_rw_mtx);
            if (sd->m_chash_map.find(service_name) != sd->m_chash_map.end())
            {
                sd->m_chash_map[service_name]->UpdateNodes(new_nodes); // 重建哈希环
            }
            sd->m_nodes_cache[service_name] = new_nodes;
        }
    }

} // namespace llm_rpc
