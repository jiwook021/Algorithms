/**
 * @file Cache.cpp
 * @brief Implementation of LruCache.
 */

#include "Cache.hpp"

#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <typename K, typename V>
LruCache<K, V>::LruCache(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("Cache capacity must be > 0");
    }
}

// ---------------------------------------------------------------------------
// Get — retrieve a value and promote it to most recently used
// ---------------------------------------------------------------------------

template <typename K, typename V>
V LruCache<K, V>::Get(const K& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        throw std::out_of_range("Key not found");
    }

    // Move the accessed item to the front (most recently used).
    auto item = *it->second;
    items_.erase(it->second);
    items_.push_front(item);
    cache_[key] = items_.begin();

    return item.second;
}

// ---------------------------------------------------------------------------
// Put — insert or update a key-value pair
// ---------------------------------------------------------------------------

template <typename K, typename V>
void LruCache<K, V>::Put(const K& key, const V& value) {
    auto it = cache_.find(key);

    if (it != cache_.end()) {
        // Key already exists: remove the old entry so we can
        // re-insert at the front with the updated value.
        items_.erase(it->second);
        cache_.erase(it);
    } else if (cache_.size() >= capacity_) {
        // Cache is full: evict the item at the back of the list
        // (the one that hasn't been used for the longest time).
        auto last = items_.back();
        cache_.erase(last.first);
        items_.pop_back();
    }

    // Insert the new item at the front (most recently used).
    items_.push_front(std::make_pair(key, value));
    cache_[key] = items_.begin();
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

template <typename K, typename V>
bool LruCache<K, V>::Exists(const K& key) const {
    return cache_.find(key) != cache_.end();
}

template <typename K, typename V>
std::size_t LruCache<K, V>::Size() const {
    return cache_.size();
}

template <typename K, typename V>
std::size_t LruCache<K, V>::Capacity() const {
    return capacity_;
}

template <typename K, typename V>
std::vector<std::pair<K, V>> LruCache<K, V>::GetItems() const {
    return std::vector<std::pair<K, V>>(items_.begin(), items_.end());
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template class LruCache<int, int>;
template class LruCache<std::string, int>;
template class LruCache<std::string, std::string>;
