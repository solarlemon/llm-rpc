#include "zookeeper_client.h"
#include "rpc_application.h" // 读配置（zookeeperip / zookeeperport）
#include <mutex>
#include "rpc_logger.h"
#include <condition_variable>

namespace llm_rpc
{

    // 全局同步原语：用于 Start() 阻塞等待"ZK 会话连接成功"
    // ------------------------------------------------------------
    // zookeeper_init 是异步的：函数返回时只是拿到了 handle，
    // 真正的 TCP 连接 + 会话建立是在 ZK 客户端后台线程里完成的。
    // 连接成功后 ZK 会回调 global_watcher，watcher 里置位 + 唤醒。
    std::mutex cv_mutex;
    std::condition_variable cv;
    bool is_connected = false; // 标志：ZK 会话是否已连接成功

    // 全局 watcher：ZK 事件通知回调
    // ------------------------------------------------------------
    // 由 ZK 客户端线程调用，不能阻塞太久。
    // 这里只关心"会话事件 + 连接成功"，置标志并唤醒 Start()。
    void global_watcher(zhandle_t *zh, int type, int status,
                        const char *path, void *watcherCtx)
    {
        if (type == ZOO_SESSION_EVENT)
        { // 会话级事件（连接/断开/过期）
            if (status == ZOO_CONNECTED_STATE)
            { // 连接成功
                std::lock_guard<std::mutex> lock(cv_mutex);
                is_connected = true;
            }
        }
        cv.notify_all(); // 唤醒等待者（Start 里的 cv.wait）
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

        // session_timeout = 6000ms；watcher = global_watcher
        m_zhandle = zookeeper_init(connstr.c_str(), global_watcher, 6000,
                                   nullptr, nullptr, 0);
        if (nullptr == m_zhandle)
        {
            LOG(ERROR) << "zookeeper_init error";
            exit(EXIT_FAILURE); // 连接参数错，直接退出
        }

        // 阻塞等待 global_watcher 置 is_connected = true
        // 谓词方式（cv.wait(lock, pred)）自动处理虚假唤醒，因为zookeeper_init 立刻返回 zhandle_t*，此时 TCP 连接还没建立。
        std::unique_lock<std::mutex> lock(cv_mutex);
        cv.wait(lock, []
                { return is_connected; });
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