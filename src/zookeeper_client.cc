#include "zookeeper_client.h"
#include "rpc_application.h" // 读配置（zookeeperip / zookeeperport）
#include <mutex>
#include "rpc_logger.h"
#include <chrono>
#include <stdexcept>

namespace llm_rpc
{

    void ZkClient::WatchSession(zhandle_t *, int type, int state, const char *, void *context)
    {
        if (type != ZOO_SESSION_EVENT)
            return;
        auto &client = *static_cast<ZkClient *>(context);
        {
            std::lock_guard<std::mutex> lock(client.session_mutex_);
            client.connected_ = state == ZOO_CONNECTED_STATE;
        }
        client.session_changed_.notify_all();
    }

    // 构造 / 析构
    ZkClient::ZkClient() : m_zhandle(nullptr) {}

    ZkClient::~ZkClient()
    {
        if (m_zhandle != nullptr)
        {
            zookeeper_close(m_zhandle); // 关闭连接（会删除该会话的临时节点）
        }
    }

    // Start：连接 ZooKeeper，并同步等待会话建立
    // ------------------------------------------------------------
    // 关键：zookeeper_init 是异步的！它立刻返回 handle，
    // 真正的连接成功要等 global_watcher 收到 ZOO_CONNECTED_STATE。
    // 所以这里用 条件变量 + 谓词 阻塞，直到连接成功才返回。
    void ZkClient::Start()
    {
        std::string host = RpcApplication::GetInstance().GetConfig().Load("zookeeperip");
        std::string port = RpcApplication::GetInstance().GetConfig().Load("zookeeperport");
        std::string connstr = host + ":" + port;

        m_zhandle = zookeeper_init(connstr.c_str(), WatchSession, 6000, nullptr, this, 0);
        if (!m_zhandle)
            throw std::runtime_error("zookeeper_init failed");

        std::unique_lock<std::mutex> lock(session_mutex_);
        if (!session_changed_.wait_for(lock, std::chrono::seconds(2), [this]
                                       { return connected_; }))
        {
            // 超时保护
            lock.unlock();
            zookeeper_close(m_zhandle);
            m_zhandle = nullptr;
            throw std::runtime_error("ZooKeeper connection deadline exceeded");
        }
        LOG(INFO) << "zookeeper_init success";
    }

    // Create：若 znode 不存在则创建
    // ------------------------------------------------------------
    // state 参数决定节点类型：
    //   0             → 持久节点（服务名父节点用它）
    //   ZOO_EPHEMERAL → 临时节点（服务实例用它，会话断开自动删除）
    void ZkClient::Create(const char *path, const char *data, int datalen, int state)
    {
        char path_buffer[128];
        int bufferlen = sizeof(path_buffer);

        // 先查是否已存在（避免重复创建报 ZNODEEXISTS）
        int flag = zoo_exists(m_zhandle, path, 0, nullptr);
        if (flag == ZNONODE)
        { // 不存在 → 创建
            flag = zoo_create(m_zhandle, path, data, datalen,
                              &ZOO_OPEN_ACL_UNSAFE, // 无 ACL 限制
                              state,                // 节点类型（持久/临时）
                              path_buffer, bufferlen);
            if (flag == ZOK)
            {
                LOG(INFO) << "znode create success... path:" << path;
            }
            else
            {
                LOG(ERROR) << "znode create failed... path:" << path;
                exit(EXIT_FAILURE); // 注册失败即服务不可用，退出
            }
        }
        // 已存在则静默跳过（服务端多实例、或重启时常见）
    }

    // GetData：读取某 znode 的数据（如 "ip:port"）
    std::string ZkClient::GetData(const char *path)
    {
        char buf[64] = {0};
        int bufferlen = sizeof(buf);

        int flag = zoo_get(m_zhandle, path, 0, buf, &bufferlen, nullptr);
        if (flag != ZOK)
        {
            LOG(ERROR) << "zoo_get error";
            return "";
        }
        return buf;
    }

    // GetChildren：列出子节点，并读取每个子节点的数据
    // ------------------------------------------------------------
    // 客户端服务发现用：
    //   path = "/UserServiceRpc"
    //   子节点 = ["127.0.0.1:8000", "127.0.0.1:8001"]
    //   读取每个子节点数据 → 得到 "ip:port" 列表
    //
    // 参数 fn / cbContext：可注册 watcher，节点变化时回调（实现服务列表动态更新）
    std::vector<std::string> ZkClient::GetChildren(const char *path,
                                                   watcher_fn fn, void *cbContext)
    {
        struct String_vector nodes;

        // zoo_wget_children = 获取子节点 + 注册 watcher（子节点变化时触发 fn）
        int flag = zoo_wget_children(m_zhandle, path, fn, cbContext, &nodes);

        std::vector<std::string> vec;
        if (flag == ZOK)
        {
            for (int i = 0; i < nodes.count; ++i)
            {
                std::string child_path = std::string(path);
                if (child_path.back() != '/')
                    child_path += "/";
                child_path += nodes.data[i];

                std::string data = GetData(child_path.c_str()); // 读子节点数据
                if (!data.empty())
                {
                    vec.push_back(data);
                }
            }
            deallocate_String_vector(&nodes); // 必须释放 ZK 分配的内存
        }
        else
        {
            LOG(ERROR) << "zoo_wget_children error, path: " << path;
        }

        return vec;
    }

} // namespace llm_rpc