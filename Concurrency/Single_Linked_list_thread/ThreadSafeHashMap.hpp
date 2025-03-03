/**
 * @file ThreadSafeHashMap.hpp
 * @brief Thread-safe hash map with bucket-level read-write locks.
 *
 * @details
 * Uses fine-grained locking (one shared_mutex per bucket) to allow
 * concurrent reads and exclusive writes. Rehashes automatically when
 * the load factor exceeds the configured threshold.
 *
 * Complexity (average case):
 *   - Insert(): O(1)
 *   - Find():   O(1)
 *   - Erase():  O(1)
 */

#ifndef THREAD_SAFE_HASHMAP_HPP
#define THREAD_SAFE_HASHMAP_HPP

#include <vector>
#include <list>
#include <optional>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <stdexcept>

/**
 * @class ThreadSafeHashMap
 * @brief Hash map with per-bucket shared_mutex for concurrent access.
 * @tparam K Key type.
 * @tparam V Value type.
 * @tparam Hash Hash function type (defaults to std::hash<K>).
 */
template <typename K, typename V, typename Hash = std::hash<K>>
class ThreadSafeHashMap {
private:
    /** @brief Key-value pair stored in bucket chains. */
    struct KeyValuePair {
        K key;     ///< The key.
        V value;   ///< The value.
        /** @brief Construct a key-value pair. */
        KeyValuePair(const K& k, const V& v) : key(k), value(v) {}
    };

    size_t numBuckets_;                                  ///< Current bucket count.
    std::vector<std::list<KeyValuePair>> buckets_;       ///< Bucket chains.
    mutable std::vector<std::shared_mutex> bucketMutexes_; ///< Per-bucket locks.
    Hash hashFunc_;                                      ///< Hash function.
    std::atomic<size_t> elementCount_;                    ///< Total element count.
    float maxLoadFactor_;                                 ///< Rehash threshold.

    /**
     * @brief Compute the bucket index for a key.
     * @param key The key to hash.
     * @return Bucket index.
     */
    size_t GetBucketIndex(const K& key) const {
        return hashFunc_(key) % numBuckets_;
    }

    /**
     * @brief Redistribute elements into a new, larger bucket array.
     * @note Caller must hold all bucket locks.
     */
    void Rehash() {
        auto oldBuckets = std::move(buckets_);
        auto oldNumBuckets = numBuckets_;
        numBuckets_ *= 2;
        buckets_.resize(numBuckets_);
        elementCount_ = 0;
        for (size_t i = 0; i < oldNumBuckets; ++i) {
            for (auto& kv : oldBuckets[i]) {
                size_t idx = hashFunc_(kv.key) % numBuckets_;
                buckets_[idx].emplace_back(kv.key, kv.value);
                ++elementCount_;
            }
        }
    }

public:
    /**
     * @brief Construct a ThreadSafeHashMap.
     * @param initialBuckets Initial number of buckets (default 16).
     * @param loadFactor Maximum load factor before rehash (default 0.75).
     * @throws std::invalid_argument if buckets is 0 or loadFactor out of (0,1).
     */
    explicit ThreadSafeHashMap(size_t initialBuckets = 16, float loadFactor = 0.75f)
        : numBuckets_(initialBuckets),
          buckets_(initialBuckets),
          bucketMutexes_(initialBuckets),
          elementCount_(0),
          maxLoadFactor_(loadFactor)
    {
        if (initialBuckets == 0)
            throw std::invalid_argument("Number of buckets must be greater than 0");
        if (loadFactor <= 0.0f || loadFactor >= 1.0f)
            throw std::invalid_argument("Load factor must be between 0 and 1");
    }

    /** @brief Non-copyable. */
    ThreadSafeHashMap(const ThreadSafeHashMap&) = delete;
    /** @brief Non-copyable. */
    ThreadSafeHashMap& operator=(const ThreadSafeHashMap&) = delete;

    /**
     * @brief Insert or update a key-value pair.
     * @param key The key.
     * @param value The value.
     */
    void Insert(const K& key, const V& value) {
        size_t bucketIdx = GetBucketIndex(key);
        std::unique_lock<std::shared_mutex> lock(bucketMutexes_[bucketIdx]);

        auto& bucket = buckets_[bucketIdx];
        for (auto& kv : bucket) {
            if (kv.key == key) { kv.value = value; return; }
        }

        bucket.emplace_back(key, value);
        size_t newCount = ++elementCount_;

        float currentLf = static_cast<float>(newCount) / numBuckets_;
        if (currentLf > maxLoadFactor_) {
            lock.unlock();
            std::vector<std::unique_lock<std::shared_mutex>> allLocks;
            allLocks.reserve(numBuckets_);
            for (size_t i = 0; i < numBuckets_; ++i)
                allLocks.emplace_back(bucketMutexes_[i]);
            Rehash();
        }
    }

    /**
     * @brief Find a value by key.
     * @param key The key to search for.
     * @return The value if found, std::nullopt otherwise.
     */
    std::optional<V> Find(const K& key) const {
        size_t bucketIdx = GetBucketIndex(key);
        std::shared_lock<std::shared_mutex> lock(bucketMutexes_[bucketIdx]);
        const auto& bucket = buckets_[bucketIdx];
        for (const auto& kv : bucket) {
            if (kv.key == key) return kv.value;
        }
        return std::nullopt;
    }

    /**
     * @brief Erase a key-value pair.
     * @param key The key to remove.
     * @return True if the key was found and removed.
     */
    bool Erase(const K& key) {
        size_t bucketIdx = GetBucketIndex(key);
        std::unique_lock<std::shared_mutex> lock(bucketMutexes_[bucketIdx]);
        auto& bucket = buckets_[bucketIdx];
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (it->key == key) {
                bucket.erase(it);
                --elementCount_;
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Return the number of elements.
     * @return Element count.
     */
    size_t Size() const { return elementCount_.load(); }

    /**
     * @brief Check if the map is empty.
     * @return True if empty.
     */
    bool Empty() const { return elementCount_.load() == 0; }

    /**
     * @brief Remove all elements.
     */
    void Clear() {
        std::vector<std::unique_lock<std::shared_mutex>> allLocks;
        allLocks.reserve(numBuckets_);
        for (size_t i = 0; i < numBuckets_; ++i)
            allLocks.emplace_back(bucketMutexes_[i]);
        for (auto& bucket : buckets_) bucket.clear();
        elementCount_ = 0;
    }

    /**
     * @brief Return the current load factor.
     * @return Load factor (elements / buckets).
     */
    float LoadFactor() const {
        return static_cast<float>(elementCount_.load()) / numBuckets_;
    }
};

#endif // THREAD_SAFE_HASHMAP_HPP
