/**
 * @file UnsafeQueue.hpp
 * @brief Intentionally UNSAFE producer-consumer queue -- demonstrates data races.
 *
 * @details
 * This queue has NO mutex protecting the std::queue. Multiple threads
 * accessing Produce()/Consume() concurrently will exhibit undefined behavior.
 * It exists purely as a teaching tool to contrast with the safe version.
 */

#ifndef UNSAFE_QUEUE_HPP
#define UNSAFE_QUEUE_HPP

#include <queue>
#include <semaphore>
#include <iostream>

/**
 * @class UnsafeQueue
 * @brief Queue with intentional data races for educational purposes.
 *
 * The semaphore tracks item availability, but the underlying std::queue
 * has no mutex protection -- concurrent access is undefined behavior.
 */
class UnsafeQueue {
public:
    /** @brief Construct an empty unsafe queue. */
    explicit UnsafeQueue() : availableItems_(0) {}

    /** @brief Non-copyable. */
    UnsafeQueue(const UnsafeQueue&) = delete;
    /** @brief Non-copyable. */
    UnsafeQueue& operator=(const UnsafeQueue&) = delete;

    /**
     * @brief Push an item without any lock -- intentional data race.
     * @param item The item to enqueue.
     */
    void Produce(int item) {
        queue_.push(item);
        std::cout << "Produced: " << item << "\n";
        availableItems_.release();
    }

    /**
     * @brief Pop an item without any lock -- intentional data race.
     * @param[out] out The dequeued item.
     * @return True if an item was dequeued.
     */
    bool Consume(int& out) {
        availableItems_.acquire();
        if (!queue_.empty()) {
            out = queue_.front();
            queue_.pop();
            std::cout << "Consumed: " << out << "\n";
            return true;
        }
        std::cout << "Entered Consume (empty)\n";
        return false;
    }

    /**
     * @brief Check if the queue is empty (not thread-safe).
     * @return True if empty.
     */
    bool Empty() const { return queue_.empty(); }

private:
    std::counting_semaphore<> availableItems_;  ///< Item availability tracker.
    std::queue<int> queue_;                      ///< Unprotected queue.
};

#endif // UNSAFE_QUEUE_HPP
