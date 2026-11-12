#ifndef LLM_RPC_ZOOKEEPER_CLIENT_H
#define LLM_RPC_ZOOKEEPER_CLIENT_H

#include <condition_variable>
#include <mutex>
#include <zookeeper/zookeeper.h>
#include <vector>
#include <string>

namespace llm_rpc
{
    // 封装的zk客户端
    class ZkClient
    {
    public:
        ZkClient();
        ~ZkClient();
        // zkclient启动连接zkserver
        void Start();
        // 在zkserver中创建一个节点，根据指定的path
        void Create(const char *path, const char *data, int datalen, int state = 0);
        // 根据参数指定的znode节点路径，或者znode节点值
        std::string GetData(const char *path);
        std::vector<std::string> GetChildren(const char *path, watcher_fn fn, void *cbContext);

    private:
        // Zk的客户端句柄
        static void WatchSession(zhandle_t *, int type, int state, const char *, void *context);
        zhandle_t *m_zhandle;
        // 标志位、互斥锁、条件变量都是成员变量，防止同一个进程中多个zkclient实例之间的干扰
        std::mutex session_mutex_;
        std::condition_variable session_changed_;
        bool connected_ = false;
    };

} // namespace llm_rpc

#endif
