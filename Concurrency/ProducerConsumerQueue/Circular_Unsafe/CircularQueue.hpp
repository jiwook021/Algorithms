/**
 * @file CircularQueue.hpp
 * @brief Fixed-size circular buffer and semaphore-based safe wrapper.
 *
 * @details
 * CircularQueue: a plain (non-thread-safe) ring buffer with O(1) enqueue/dequeue.
 * SafeQueue: wraps CircularQueue with counting semaphores so producers
 * block when full and consumers block when empty.
 */

#ifndef CIRCULAR_QUEUE_HPP
#define CIRCULAR_QUEUE_HPP

#include <vector>
#include <mutex>
#include <semaphore>
#include <iostream>

/**
 * @class CircularQueue
 * @brief Fixed-size ring buffer (NOT thread-safe).
 *
 * Elements are stored in a pre-allocated vector. Head and tail indices
 * wrap around using modular arithmetic.
 */
class CircularQueue {
public:
    /**
     * @brief Construct a circular queue with given capacity.
     * @param size Maximum number of elements.
     */
    explicit CircularQueue(size_t size)
        : size_(size), head_(0), tail_(0), count_(0), data_(size) {}

    /**
     * @brief Enqueue an item.
     * @param item The item to enqueue.
     * @return True if successful, false if full.
     */
    bool Enqueue(int item) {
        if (count_ == size_) return false;
        data_[tail_] = item;
        tail_ = (tail_ + 1) % size_;
        ++count_;
        return true;
    }

    /**
     * @brief Dequeue an item.
     * @param[out] item The dequeued item.
     * @return True if successful, false if empty.
     */
    bool Dequeue(int& item) {
        if (count_ == 0) return false;
        item = data_[head_];
        head_ = (head_ + 1) % size_;
        --count_;
        return true;
    }

    /** @brief Return the current element count. */
    size_t Count() const { return count_; }

    /** @brief Return the maximum capacity. */
    size_t Capacity() const { return size_; }

    /** @brief Check if empty. */
    bool Empty() const { return count_ == 0; }

    /** @brief Check if full. */
    bool Full() const { return count_ == size_; }

private:
    const size_t size_;       ///< Maximum capacity.
    size_t head_;             ///< Read index.
    size_t tail_;             ///< Write index.
    size_t count_;            ///< Current element count.
    std::vector<int> data_;   ///< Pre-allocated storage.
};

/**
 * @class SafeQueue
 * @brief Thread-safe wrapper around CircularQueue using counting semaphores.
 *
 * Producers block when the buffer is full; consumers block when empty.
 */
class SafeQueue {
public:
    /**
     * @brief Construct a SafeQueue with given capacity.
     * @param maxSize Maximum number of elements.
     */
    explicit SafeQueue(size_t maxSize)
        : queue_(maxSize), availableItems_(0), availableSlots_(maxSize) {}

    /** @brief Non-copyable. */
    SafeQueue(const SafeQueue&) = delete;
    /** @brief Non-copyable. */
    SafeQueue& operator=(const SafeQueue&) = delete;

    /**
     * @brief Produce an item. Blocks if full.
     * @param item The item to enqueue.
     */
    void Produce(int item) {
        availableSlots_.acquire();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.Enqueue(item);
        }
        std::cout << "Produced: " << item << "\n";
        availableItems_.release();
    }

    /**
     * @brief Consume an item. Blocks if empty.
     * @param[out] out The dequeued item.
     * @return True if an item was dequeued.
     */
    bool Consume(int& out) {
        availableItems_.acquire();
        bool ok;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ok = queue_.Dequeue(out);
        }
        if (ok) std::cout << "Consumed: " << out << "\n";
        availableSlots_.release();
        return ok;
    }

private:
    CircularQueue queue_;                     ///< Underlying ring buffer.
    std::counting_semaphore<> availableItems_; ///< Tracks available items.
    std::counting_semaphore<> availableSlots_; ///< Tracks free slots.
    std::mutex mutex_;                         ///< Protects the queue.
};

#endif // CIRCULAR_QUEUE_HPP
