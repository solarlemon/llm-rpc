#ifndef LLM_RPC_SERVICE_DISCOVERY_H
#define LLM_RPC_SERVICE_DISCOVERY_H

#include "zookeeper_client.h"
#include "consistent_hash.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex> // 引入读写锁头文件

namespace llm_rpc {

class ServiceDiscovery
{
public:
    static ServiceDiscovery &GetInstance();
    void Init();
    std::string GetTargetNode(const std::string &service_name, const std::string &key);

private:
    ServiceDiscovery() = default;
    ~ServiceDiscovery() = default;
    ServiceDiscovery(const ServiceDiscovery &) = delete;
    ServiceDiscovery &operator=(const ServiceDiscovery &) = delete;

    ZkClient m_zkClient;
    std::unordered_map<std::string, std::unique_ptr<ConsistentHash>> m_chash_map; 
    std::unordered_map<std::string, std::vector<std::string>> m_nodes_cache;      
    
    
    std::shared_timed_mutex m_rw_mtx; 

    static void WatcherCallback(zhandle_t *zh, int type, int state, const char *path, void *watcherCtx);
    void UpdateCacheLocked(const std::string &service_name, const std::string &path);
};


}  // namespace llm_rpc

#endif