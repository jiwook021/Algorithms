/**
 * @file PriorityProducerConsumerQueue.cpp
 * @brief Implementation of bucketed priority PriorityProducerConsumerQueue<T>.
 */

#include "PriorityProducerConsumerQueue.hpp"

#include <stdexcept>
#include <utility>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <typename T>
PriorityProducerConsumerQueue<T>::PriorityProducerConsumerQueue(
    std::size_t max_size,
    std::size_t priority_levels)
    : max_size_(max_size),
      priority_levels_(ValidatePriorityLevels(priority_levels)),
      buckets_(priority_levels_),
      available_slots_(static_cast<std::ptrdiff_t>(max_size)),
      available_items_(0),
      priority_bucket_has_items_(priority_levels_, false) {}

template <typename T>
std::size_t PriorityProducerConsumerQueue<T>::ValidatePriorityLevels(
    std::size_t priority_levels) {
    if (priority_levels == 0 || priority_levels > kMaxPriorityLevels) {
        throw std::invalid_argument(
            "priority_levels must be between 1 and 64");
    }
    return priority_levels;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
bool PriorityProducerConsumerQueue<T>::IsValidPriority(int priority) const {
    return priority >= 0 &&
           static_cast<std::size_t>(priority) < priority_levels_;
}

template <typename T>
std::size_t PriorityProducerConsumerQueue<T>::HighestPriority() const {
    for (std::size_t priority = priority_levels_; priority > 0; --priority) {
        const std::size_t index = priority - 1;
        if (priority_bucket_has_items_[index]) {
            return index;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Produce -- acquire slot, lock, insert into bucket, release item
// ---------------------------------------------------------------------------

template <typename T>
bool PriorityProducerConsumerQueue<T>::Produce(int id, int priority, T item) {
    if (!IsValidPriority(priority)) {
        return false;
    }

    {
        std::scoped_lock lock(queue_mutex_);
        if (item_locations_by_id_.contains(id)) {
            return false;
        }
    }

    available_slots_.acquire();          // block if queue is full

    {
        std::scoped_lock lock(queue_mutex_);
        if (item_locations_by_id_.contains(id)) {
            available_slots_.release();
            return false;
        }

        auto& bucket = buckets_[static_cast<std::size_t>(priority)];
        bucket.push_back(StoredItem{id, priority, std::move(item)});

        auto inserted = std::prev(bucket.end());
        item_locations_by_id_.emplace(id, Location{priority, inserted});
        priority_bucket_has_items_[static_cast<std::size_t>(priority)] = true;
    }

    available_items_.release();          // signal waiting consumers
    return true;
}

// ---------------------------------------------------------------------------
// UpdatePriority -- move existing list node between priority buckets
// ---------------------------------------------------------------------------

template <typename T>
bool PriorityProducerConsumerQueue<T>::UpdatePriority(int id, int new_priority) {
    if (!IsValidPriority(new_priority)) {
        return false;
    }

    std::scoped_lock lock(queue_mutex_);

    auto found = item_locations_by_id_.find(id);
    if (found == item_locations_by_id_.end()) {
        return false;
    }

    auto& location = found->second;
    const int old_priority = location.priority;
    if (old_priority == new_priority) {
        return true;
    }

    auto& old_bucket = buckets_[static_cast<std::size_t>(old_priority)];
    auto& new_bucket = buckets_[static_cast<std::size_t>(new_priority)];
    auto item_it = location.it;

    item_it->priority = new_priority;

    // Move the existing list node into the new priority bucket in O(1).
    // The stored data is not copied; the node is relinked at the bucket's end.
    new_bucket.splice(new_bucket.end(), old_bucket, item_it);

    if (old_bucket.empty()) {
        priority_bucket_has_items_[static_cast<std::size_t>(old_priority)] = false;
    }
    priority_bucket_has_items_[static_cast<std::size_t>(new_priority)] = true;

    location.priority = new_priority;
    location.it = item_it;
    return true;
}

// ---------------------------------------------------------------------------
// Consume -- acquire item, lock, pop highest non-empty priority, release slot
// ---------------------------------------------------------------------------

template <typename T>
std::optional<PriorityItem<T>> PriorityProducerConsumerQueue<T>::Consume() {
    available_items_.acquire();          // block if queue is empty

    std::optional<PriorityItem<T>> item;
    {
        std::scoped_lock lock(queue_mutex_);
        if (item_locations_by_id_.empty()) {
            available_slots_.release();
            return std::nullopt;
        }

        const std::size_t priority = HighestPriority();
        auto& bucket = buckets_[priority];
        auto& front = bucket.front();

        item.emplace(PriorityItem<T>{
            front.id,
            front.priority,
            std::move(front.data),
        });

        item_locations_by_id_.erase(item->id);
        bucket.pop_front();

        if (bucket.empty()) {
            priority_bucket_has_items_[priority] = false;
        }
    }

    available_slots_.release();          // signal waiting producers
    return item;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

template <typename T>
bool PriorityProducerConsumerQueue<T>::Contains(int id) const {
    std::scoped_lock lock(queue_mutex_);
    return item_locations_by_id_.contains(id);
}

template <typename T>
std::size_t PriorityProducerConsumerQueue<T>::Size() const {
    std::scoped_lock lock(queue_mutex_);
    return item_locations_by_id_.size();
}

template <typename T>
std::size_t PriorityProducerConsumerQueue<T>::MaxSize() const {
    return max_size_;
}

template <typename T>
std::size_t PriorityProducerConsumerQueue<T>::PriorityLevels() const {
    return priority_levels_;
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template class PriorityProducerConsumerQueue<int>;
template class PriorityProducerConsumerQueue<std::string>;
template class PriorityProducerConsumerQueue<std::unique_ptr<int>>;
