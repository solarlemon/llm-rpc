#include "rpc_application.h"
#include "../user.pb.h"
#include "rpc_controller.h"
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include "rpc_logger.h"
#include "rpc_connection_pool.h"

void send_request(int thread_id, std::atomic<int> &success_count, std::atomic<int> &fail_count, int requests_per_thread)
{
    // 注意这里增加了 STUB_OWNS_CHANNEL 自动管理 Channel 内存
    llm_rpc::example::UserServiceRpc_Stub stub(new llm_rpc::RpcChannel(false), google::protobuf::Service::STUB_OWNS_CHANNEL);

    llm_rpc::example::LoginRequest request;
    request.set_name("zhangsan");
    request.set_pwd("123456");

    llm_rpc::example::LoginResponse response;
    llm_rpc::RpcController controller;

    for (int i = 0; i < requests_per_thread; ++i)
    {
        // 【关键修复 1】：每次请求前必须重置控制器，否则一次失败会导致后续全盘误判为失败！
        controller.Reset();

        stub.Login(&controller, &request, &response, nullptr);

        if (controller.Failed())
        {
            // 失败时不要频繁打印 cout，这会极其严重地拖慢多线程性能，改成计数即可
            // std::cout << controller.ErrorText() << std::endl;
            fail_count++;
        }
        else
        {
            if (int{} == response.result().errcode())
            {
                success_count++;
            }
            else
            {
                fail_count++;
            }
        }
    }
}

int main(int argc, char **argv)
{
    llm_rpc::RpcApplication::Init(argc, argv);
    FLAGS_logbufsecs = 5;
    llm_rpc::RpcLogger logger("LLM-RPC");
    std::string ip = llm_rpc::RpcApplication::GetConfig().Load("rpcserverip");
    uint16_t port = atoi(llm_rpc::RpcApplication::GetConfig().Load("rpcserverport").c_str());

    // 预热 100 个长连接
    const int thread_count = 100;
    llm_rpc::RpcConnectionPool::GetInstance().WarmUp(ip, port, thread_count);

    const int requests_per_thread = 1000;

    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);
    std::atomic<int> fail_count(0);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < thread_count; i++)
    {
        threads.emplace_back([argc, argv, i, &success_count, &fail_count, requests_per_thread]()
                             { send_request(i, success_count, fail_count, requests_per_thread); });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    LOG(INFO) << "Total requests: " << thread_count * requests_per_thread;
    LOG(INFO) << "Success count: " << success_count;
    LOG(INFO) << "Fail count: " << fail_count;
    LOG(INFO) << "Elapsed time: " << elapsed.count() << " seconds";
    LOG(INFO) << "QPS: " << (thread_count * requests_per_thread) / elapsed.count();

    return 0;
}