#ifndef LLM_RPC_LOCK_FREE_QUEUE_H
#define LLM_RPC_LOCK_FREE_QUEUE_H

#include <atomic>
#include <vector>
#include <stdexcept>

namespace llm_rpc
{

    template <typename T>
    class MPMCQueue
    {
    public:
        // capacity 必须是 2 的幂：这样 pos & mask 等价于 pos % capacity，
        // 位运算比取模快很多，且无除法指令
        explicit MPMCQueue(size_t capacity)
            : capacity_(capacity), mask_(capacity - 1), buffer_(capacity)
        {
            // 校验 2 的幂：capacity & (capacity-1) 对 2 的幂必为 0
            if ((capacity & mask_) != 0)
            {
                throw std::invalid_argument("Capacity must be a power of 2");
            }
            // 初始化每个格子的 sequence = 下标 i，表示"第 i 格空，等位置 i 来写"
            for (size_t i = 0; i < capacity_; ++i)
            {
                buffer_[i].sequence.store(i, std::memory_order_relaxed);
            }
            enqueue_pos_.store(0, std::memory_order_relaxed);
            dequeue_pos_.store(0, std::memory_order_relaxed);
        }

        // push：入队（生产者）
        bool push(T const &data)
        {
            Cell *cell;
            size_t pos = enqueue_pos_.load(std::memory_order_relaxed);

            for (;;)
            {
                cell = &buffer_[pos & mask_];                                // 定位环形位置
                size_t seq = cell->sequence.load(std::memory_order_acquire); // 读该格状态
                // diff = 该格实际 seq - 期望 seq（期望 = pos）
                intptr_t diff = (intptr_t)seq - (intptr_t)pos;

                if (diff == 0)
                {
                    // 该格空着，可写 → 尝试抢占 enqueue_pos（CAS）
                    // compare_exchange_weak：失败会把最新值写回 pos，供下轮重试
                    if (enqueue_pos_.compare_exchange_weak(pos, pos + 1,
                                                           std::memory_order_relaxed))
                        break; // 抢占成功，跳出循环去写数据
                    // 抢占失败：pos 已被更新为最新值，继续下一轮
                }
                else if (diff < 0)
                {
                    // seq < pos：该格还是上一轮的数据（没被消费），队列满
                    return false;
                }
                else
                {
                    // seq > pos：pos 读旧了（其他生产者已推进），重读
                    pos = enqueue_pos_.load(std::memory_order_relaxed);
                }
            }

            // 抢占成功，写入数据
            cell->data = data;
            // release：保证 data 写入对其他线程可见后，才发布 seq = pos+1
            cell->sequence.store(pos + 1, std::memory_order_release);
            return true;
        }

        // pop：出队（消费者）
        bool pop(T &data)
        {
            Cell *cell;
            size_t pos = dequeue_pos_.load(std::memory_order_relaxed);

            for (;;)
            {
                cell = &buffer_[pos & mask_];
                size_t seq = cell->sequence.load(std::memory_order_acquire);
                // 消费者期望：seq == pos + 1（表示该格已有数据）
                intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);

                if (diff == 0)
                {
                    // 该格有数据，可读 → 尝试抢占 dequeue_pos
                    if (dequeue_pos_.compare_exchange_weak(pos, pos + 1,
                                                           std::memory_order_relaxed))
                        break; // 抢占成功，跳出循环去读数据
                }
                else if (diff < 0)
                {
                    // seq < pos+1：该格还没被生产（生产者没写），队列空
                    return false;
                }
                else
                {
                    // seq > pos+1：pos 读旧了，重读
                    pos = dequeue_pos_.load(std::memory_order_relaxed);
                }
            }

            // 抢占成功，读取数据
            data = cell->data;
            // 标记该格已消费：seq = pos + capacity（等价于回到"空"状态，供下一轮位置写）
            // release：保证 data 被读完后再发布状态
            cell->sequence.store(pos + mask_ + 1, std::memory_order_release);
            return true;
        }

    private:
        struct Cell
        {
            std::atomic<size_t> sequence; // 状态标记：控制该格空/满
            T data;                       // 实际数据（如 int fd）
        };

        const size_t capacity_;    // 容量（2 的幂）
        const size_t mask_;        // capacity - 1，用于 & 取模
        std::vector<Cell> buffer_; // 环形缓冲区

        // 强制缓存行对齐（64 字节），防止"伪共享 (False Sharing)"：
        // 若 enqueue_pos 和 dequeue_pos 落在同一缓存行，
        // 生产者和消费者各改一个，会互相使对方缓存行失效，严重拖慢性能
        alignas(64) std::atomic<size_t> enqueue_pos_;
        alignas(64) std::atomic<size_t> dequeue_pos_;
    };

} // namespace llm_rpc

#endif