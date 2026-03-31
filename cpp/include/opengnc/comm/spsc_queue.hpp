#pragma once

#include <atomic>
#include <cassert>
#include <vector>

namespace opengnc {
namespace comm {

/**
 * Single-Producer Single-Consumer (SPSC) Lock-Free Queue.
 * Optimized for deterministic real-time communication between high-priority GNC tasks
 * and low-priority telemetry/logging tasks.
 *
 * @tparam T The type of data to store in the queue.
 * @tparam Capacity Total size of the ring buffer (must be a power of 2 for optimization).
 */
template <typename T, size_t Capacity>
class SPSCQueue {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2 for efficient indexing.");

    SPSCQueue() : head(0), tail(0) {}

    /**
     * Push an item into the queue.
     * @param data The item to push.
     * @return true if successful, false if the queue is full.
     */
    bool push(const T& data) {
        const size_t current_tail = tail.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask;
        
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        buffer[current_tail] = data;
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    /**
     * Pop an item from the queue.
     * @param result Reference to store the popped item.
     * @return true if an item was popped, false if the queue is empty.
     */
    bool pop(T& result) {
        const size_t current_head = head.load(std::memory_order_relaxed);
        
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }
        
        result = buffer[current_head];
        head.store((current_head + 1) & mask, std::memory_order_release);
        return true;
    }

    /**
     * Check if the queue is empty.
     */
    bool empty() const {
        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
    }

    /**
     * Estimate of the current number of items (not perfectly reliable due to concurrency).
     */
    size_t size() const {
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_relaxed);
        if (t >= h) return t - h;
        return Capacity - (h - t);
    }

private:
    T buffer[Capacity];
    static constexpr size_t mask = Capacity - 1;

    // Use alignment to prevent false sharing if head/tail are updated by different cores
    alignas(64) std::atomic<size_t> head;
    alignas(64) std::atomic<size_t> tail;
};

} // namespace comm
} // namespace opengnc
