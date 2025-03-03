/**
 * @file Cache.hpp
 * @brief LRU (Least Recently Used) cache.
 *
 * @details
 * A cache stores recently used data for fast retrieval.  When the
 * cache is full, the item that hasn't been used for the longest
 * time gets evicted to make room for new data.
 *
 * How it works:
 *   - A doubly-linked list tracks usage order.  The front of the
 *     list is the most recently used item; the back is the least
 *     recently used.
 *   - A hash map maps each key to its position in the list, giving
 *     O(1) lookup, insertion, and eviction.
 *   - Get(key) moves the item to the front (most recent).
 *   - Put(key, value) adds a new item at the front.  If the cache
 *     is already full, the item at the back (least recently used)
 *     is removed first.
 *
 * Complexity:
 *   Get  : O(1)
 *   Put  : O(1)
 *   Space: O(capacity)
 */

#pragma once

#include <cstddef>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @class LruCache
 * @brief Key-value cache that evicts the least recently used item when full.
 *
 * @tparam K Key type (must be hashable via std::hash).
 * @tparam V Value type.
 */
template <typename K, typename V>
class LruCache {
public:
    /**
     * @brief Create a cache that holds at most @p capacity items.
     * @throws std::invalid_argument if capacity is 0.
     */
    explicit LruCache(std::size_t capacity);

    /**
     * @brief Look up a key and return its value.
     *
     * Moves the item to "most recently used" so it will be
     * the last to be evicted.
     *
     * @throws std::out_of_range if the key is not in the cache.
     */
    V Get(const K& key);

    /**
     * @brief Store a key-value pair in the cache.
     *
     * If the key already exists, its value is updated and it moves
     * to most recently used.  If the cache is full, the least
     * recently used item is evicted first.
     */
    void Put(const K& key, const V& value);

    /** @brief Check whether a key is in the cache (does not change usage order). */
    [[nodiscard]] bool Exists(const K& key) const;

    /** @brief Return the number of items currently in the cache. */
    [[nodiscard]] std::size_t Size() const;

    /** @brief Return the maximum number of items the cache can hold. */
    [[nodiscard]] std::size_t Capacity() const;

    /**
     * @brief Return all cached items in usage order (most recent first).
     */
    [[nodiscard]] std::vector<std::pair<K, V>> GetItems() const;

private:
    std::size_t capacity_;
    std::list<std::pair<K, V>> items_;
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> cache_;
};
