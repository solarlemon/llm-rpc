#include "rpc_application.h"
#include "../user.pb.h"
#include "rpc_controller.h"
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include "rpc_logger.h"
#include "rpc_connection_pool.h"

// 单个线程的执行体：循环发 requests_per_thread 次 Login 请求
void send_request(int thread_id, std::atomic<int> &success_count, std::atomic<int> &fail_count, int requests_per_thread)
{
    // 创建 Stub（客户端桩），底层传入自定义 RpcChannel
    // 第二个参数 STUB_OWNS_CHANNEL：让 Stub 析构时自动 delete 掉 channel，
    // 避免每线程手动管理内存造成泄漏
    llm_rpc::example::UserServiceRpc_Stub stub(
        new llm_rpc::RpcChannel(false), // 造一个"传输实现" RpcChannel。不立即连接，等到第一次调用时再连接
        // 线程结束时 stub 析构 → 自动 delete channel → 无泄漏
        google::protobuf::Service::STUB_OWNS_CHANNEL); // 告诉 Stub 接管它的生命周期

    // 构造请求参数（所有请求共用，只读，线程内复用节省开销）
    llm_rpc::example::LoginRequest request;
    request.set_name("zhangsan");
    request.set_pwd("123456");
    
    // 响应和控制器对象在循环外复用
    llm_rpc::example::LoginResponse response;
    llm_rpc::RpcController controller;

    for (int i = 0; i < requests_per_thread; ++i)
    {
        // 【关键修复 1】：每次请求前必须重置控制器，否则一次失败会导致后续全盘误判为失败！
        controller.Reset();

        // 发起 RPC 调用（同步阻塞直到返回响应），调用channel的CallMethod方法，底层会通过socket发送请求到服务端
        // 参数：controller、request、response、done 回调（nullptr = 同步等待）
        stub.Login(&controller, &request, &response, nullptr);

        if (controller.Failed())
        {
            // 失败时不要频繁打印 cout，这会极其严重地拖慢多线程性能，改成计数即可
            // std::cout << controller.ErrorText() << std::endl;
            fail_count++;
        }
        else
        {
            if (int{} == response.result().errcode())  // errcode == 0 表示业务成功
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
    // 1. 初始化 RPC 框架（解析命令行、加载配置文件等）
    llm_rpc::RpcApplication::Init(argc, argv);
    FLAGS_logbufsecs = 5; // glog 缓冲 5 秒刷盘，减少 IO
    llm_rpc::RpcLogger logger("LLM-RPC"); // 初始化日志器

    // 2. 从配置文件读取服务端地址
    std::string ip = llm_rpc::RpcApplication::GetConfig().Load("rpcserverip"); 
    uint16_t port = atoi(llm_rpc::RpcApplication::GetConfig().Load("rpcserverport").c_str());

    // 3. 预热连接池：提前建立 100 条长连接，避免压测时把建连时间算进来
    const int thread_count = 100;
    llm_rpc::RpcConnectionPool::GetInstance().WarmUp(ip, port, thread_count);

    const int requests_per_thread = 1000; // 线程数 == 预热连接数

    std::vector<std::thread> threads;                      // 线程容器
    std::atomic<int> success_count(0);                     // 成功计数（无锁原子）
    std::atomic<int> fail_count(0);                        // 失败计数

    auto start_time = std::chrono::high_resolution_clock::now();

    // 4. 启动 thread_count 个线程，各自压测
    for (int i = 0; i < thread_count; i++)
    {
        // 按值捕获 argc/argv/i/requests_per_thread，按引用捕获两个原子计数
        threads.emplace_back([argc, argv, i, &success_count, &fail_count, requests_per_thread]()
                             { send_request(i, success_count, fail_count, requests_per_thread); });
    }

    // 6. 等待所有线程结束
    for (auto &t : threads)
    {
        t.join();
    }

    // 7. 结束计时，输出压测结果
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    LOG(INFO) << "Total requests: " << thread_count * requests_per_thread;
    LOG(INFO) << "Success count: " << success_count;
    LOG(INFO) << "Fail count: " << fail_count;
    LOG(INFO) << "Elapsed time: " << elapsed.count() << " seconds";
    LOG(INFO) << "QPS: " << (thread_count * requests_per_thread) / elapsed.count();

    return 0;
}