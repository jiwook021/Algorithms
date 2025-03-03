/**
 * @file PriorityProducerConsumerQueue.hpp
 * @brief Thread-safe bounded producer-consumer queue with bucketed priorities.
 *
 * @details
 * Producers add uniquely identified work items with a bounded priority.
 * Consumers remove the highest-priority waiting item.  Items with the same
 * priority are consumed FIFO.
 *
 * The queue uses:
 *   - one counting semaphore for free slots,
 *   - one counting semaphore for available items,
 *   - one mutex for shared state,
 *   - one FIFO bucket per priority level,
 *   - an id map so waiting items can be reprioritized in O(1).
 *
 * Priority values are 0..PriorityLevels()-1.  Larger numbers mean higher
 * priority.
 */

#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @struct PriorityItem
 * @brief Public item returned by Consume().
 */
template <typename T>
struct PriorityItem {
    int id;
    int priority;
    T data;
};

/**
 * @class PriorityProducerConsumerQueue
 * @brief Bounded, thread-safe, reprioritizable producer-consumer queue.
 *
 * Produce(id, priority, data) inserts a new unique id.
 * UpdatePriority(id, priority) moves a waiting item between priority buckets.
 * Consume() removes the oldest item from the highest non-empty priority bucket.
 *
 * The consumer path is O(1) with respect to the number of queued items because
 * the priority range is bounded to at most 64 buckets.
 *
 * @tparam T Element data type.
 */
template <typename T>
class PriorityProducerConsumerQueue {
public:
    static constexpr std::size_t kDefaultPriorityLevels = 4;
    static constexpr std::size_t kMaxPriorityLevels = 64;

    explicit PriorityProducerConsumerQueue(
        std::size_t max_size,
        std::size_t priority_levels = kDefaultPriorityLevels);

    PriorityProducerConsumerQueue(const PriorityProducerConsumerQueue&) = delete;
    PriorityProducerConsumerQueue& operator=(const PriorityProducerConsumerQueue&) = delete;

    /**
     * @brief Enqueue a new item. Blocks if the queue is full.
     * @return True if inserted, false for duplicate id or invalid priority.
     */
    bool Produce(int id, int priority, T item);

    /**
     * @brief Update the priority of a waiting item by id.
     * @return True if updated, false if id is missing or priority is invalid.
     */
    bool UpdatePriority(int id, int new_priority);

    /**
     * @brief Dequeue the highest-priority waiting item. Blocks if empty.
     */
    [[nodiscard]] std::optional<PriorityItem<T>> Consume();

    /** @brief Return true if id currently exists in the queue. */
    [[nodiscard]] bool Contains(int id) const;

    /** @brief Return the current number of live elements. */
    [[nodiscard]] std::size_t Size() const;

    /** @brief Return the maximum capacity. */
    [[nodiscard]] std::size_t MaxSize() const;

    /** @brief Return the number of available priority buckets. */
    [[nodiscard]] std::size_t PriorityLevels() const;

private:
    struct StoredItem {
        int id;
        int priority;
        T data;
    };

    struct Location {
        int priority;
        typename std::list<StoredItem>::iterator it;
    };

    static std::size_t ValidatePriorityLevels(std::size_t priority_levels);

    [[nodiscard]] bool IsValidPriority(int priority) const;
    [[nodiscard]] std::size_t HighestPriority() const;

    std::size_t max_size_;
    std::size_t priority_levels_;
    std::vector<std::list<StoredItem>> buckets_;
    // Maps each live item id to its priority bucket and exact list position.
    std::unordered_map<int, Location> item_locations_by_id_;

    // Protects the buckets, id lookup map, and priority bucket state.
    mutable std::mutex queue_mutex_;

    // Counts free capacity slots; producers wait here when the queue is full.
    std::counting_semaphore<> available_slots_;

    // Counts queued items; consumers wait here when the queue is empty.
    std::counting_semaphore<> available_items_;

    // Tracks which priority buckets currently contain at least one item.
    std::vector<bool> priority_bucket_has_items_;
};
