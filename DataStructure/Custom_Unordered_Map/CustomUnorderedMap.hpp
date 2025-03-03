/**
 * @file CustomUnorderedMap.hpp
 * @brief Hash table with separate chaining (custom std::unordered_map).
 *
 * @details
 * A hash table stores key-value pairs and provides O(1) average-time
 * lookup, insertion, and deletion.
 *
 * How it works:
 *   1. Hash the key to get a number.
 *   2. Pick a bucket: hash % number_of_buckets.
 *   3. Store the key-value pair in that bucket.
 *   4. If two keys land in the same bucket (a "collision"), both
 *      are stored in a linked list ("separate chaining").
 *   5. When the average chain length (load factor) gets too high,
 *      the table doubles its bucket count and redistributes all
 *      elements ("rehashing") to keep lookups fast.
 *
 * Complexity:
 *   Insert / Find / Erase : O(1) average, O(n) worst case
 *   Space                 : O(n + bucket_count)
 */

#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <list>
#include <utility>
#include <vector>

/**
 * @class UnorderedMap
 * @brief Key-value hash table with separate chaining and automatic rehashing.
 *
 * @tparam Key   Key type (must have a std::hash specialization).
 * @tparam Value Mapped value type.
 * @tparam Hash  Hash functor (default: std::hash<Key>).
 */
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class UnorderedMap {
private:
    struct KeyValuePair {
        Key key;
        Value value;
        KeyValuePair(const Key& k, const Value& v) : key(k), value(v) {}
    };

    std::size_t GetBucketIndex(const Key& key) const;

    typename std::list<KeyValuePair>::iterator
    FindInBucket(std::list<KeyValuePair>& bucket, const Key& key);

    typename std::list<KeyValuePair>::const_iterator
    FindInBucket(const std::list<KeyValuePair>& bucket, const Key& key) const;

    void RehashImpl(std::size_t new_bucket_count);

    std::vector<std::list<KeyValuePair>> buckets_;
    Hash hasher_;
    std::size_t element_count_;
    double max_load_factor_;

public:
    class iterator;
    class const_iterator;

    template <typename K, typename V>
    class PairProxy {
    private:
        std::pair<K, V> pair_;
    public:
        PairProxy(K k, V v) : pair_(k, v) {}
        const std::pair<K, V>* operator->() const { return &pair_; }
    };

    // ---------------------------------------------------------------
    // Construction / assignment
    // ---------------------------------------------------------------

    /**
     * @brief Create an empty map.
     * @param initial_buckets Starting number of buckets (default 16).
     * @param max_load_factor Rehash when load factor exceeds this (default 1.0).
     */
    explicit UnorderedMap(std::size_t initial_buckets = 16,
                          double max_load_factor = 1.0);

    /** @brief Construct from an initializer list of {key, value} pairs. */
    UnorderedMap(std::initializer_list<std::pair<Key, Value>> init,
                 std::size_t initial_buckets = 16,
                 double max_load_factor = 1.0);

    UnorderedMap(const UnorderedMap& other);
    UnorderedMap(UnorderedMap&& other) noexcept;
    UnorderedMap& operator=(const UnorderedMap& other);
    UnorderedMap& operator=(UnorderedMap&& other) noexcept;
    ~UnorderedMap() = default;

    // ---------------------------------------------------------------
    // Core API
    // ---------------------------------------------------------------

    /**
     * @brief Insert a key-value pair.
     * @return {iterator, true} on success, or {existing_iterator, false}
     *         if the key already exists (value is NOT overwritten).
     */
    std::pair<iterator, bool> Insert(const Key& key, const Value& value);

    /** @brief Return a pointer to the value, or nullptr if absent. */
    Value* Find(const Key& key);
    const Value* Find(const Key& key) const;

    /** @brief Return an iterator to the key, or end() if absent. */
    iterator FindIterator(const Key& key);
    const_iterator FindIterator(const Key& key) const;

    /** @brief Check whether a key exists (does not modify the map). */
    [[nodiscard]] bool Contains(const Key& key) const;

    /** @brief Erase by key.  Returns true if an element was removed. */
    bool Erase(const Key& key);

    /** @brief Erase by iterator.  Returns iterator to next element. */
    iterator Erase(const iterator& pos);

    // ---------------------------------------------------------------
    // Capacity / bucket queries
    // ---------------------------------------------------------------

    [[nodiscard]] std::size_t Size() const;
    [[nodiscard]] bool Empty() const;
    [[nodiscard]] std::size_t BucketCount() const;
    [[nodiscard]] std::size_t BucketSize(std::size_t n) const;
    [[nodiscard]] std::size_t Bucket(const Key& key) const;
    [[nodiscard]] double LoadFactor() const;
    [[nodiscard]] double MaxLoadFactor() const;
    void SetMaxLoadFactor(double ml);
    void Rehash(std::size_t count);
    void Reserve(std::size_t count);

    // ---------------------------------------------------------------
    // Element access
    // ---------------------------------------------------------------

    /** @brief Access or insert.  Creates a default value if key is new. */
    Value& operator[](const Key& key);

    /** @brief Access with bounds checking.  Throws if key is absent. */
    Value& At(const Key& key);
    const Value& At(const Key& key) const;

    /** @brief Remove all elements (bucket structure is preserved). */
    void Clear();

    // ---------------------------------------------------------------
    // Iterators
    // ---------------------------------------------------------------

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    const_iterator cbegin() const;
    const_iterator cend() const;

    // ---------------------------------------------------------------
    // STL lowercase aliases
    // ---------------------------------------------------------------

    std::pair<iterator, bool> insert(const Key& k, const Value& v) {
        return Insert(k, v);
    }
    iterator find(const Key& k) { return FindIterator(k); }
    const_iterator find(const Key& k) const { return FindIterator(k); }
    std::size_t erase(const Key& k) { return Erase(k) ? 1 : 0; }
    Value& at(const Key& k) { return At(k); }
    const Value& at(const Key& k) const { return At(k); }
    void reserve(std::size_t count) { Reserve(count); }

    // ---------------------------------------------------------------
    // Iterator classes
    //
    // Defined here because they are tightly coupled to the internal
    // bucket structure.  Each iterator holds a pointer to the map,
    // the current bucket index, and an iterator within that bucket's
    // linked list.
    // ---------------------------------------------------------------

    class iterator {
    private:
        UnorderedMap* map_;
        std::size_t bucket_index_;
        typename std::list<KeyValuePair>::iterator bucket_iterator_;
        friend class UnorderedMap;

    public:
        iterator() : map_(nullptr), bucket_index_(0) {}

        iterator(UnorderedMap* m, std::size_t bi,
                 typename std::list<KeyValuePair>::iterator it)
            : map_(m), bucket_index_(bi), bucket_iterator_(it) {}

        std::pair<const Key&, Value&> operator*() const {
            return {bucket_iterator_->key, bucket_iterator_->value};
        }

        PairProxy<const Key&, Value&> operator->() const {
            return PairProxy<const Key&, Value&>(
                bucket_iterator_->key, bucket_iterator_->value);
        }

        iterator& operator++() {
            ++bucket_iterator_;
            if (bucket_index_ < map_->buckets_.size() &&
                bucket_iterator_ == map_->buckets_[bucket_index_].end()) {
                do {
                    ++bucket_index_;
                } while (bucket_index_ < map_->buckets_.size() &&
                         map_->buckets_[bucket_index_].empty());
                if (bucket_index_ < map_->buckets_.size()) {
                    bucket_iterator_ = map_->buckets_[bucket_index_].begin();
                }
            }
            return *this;
        }

        iterator operator++(int) {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const iterator& other) const {
            if (bucket_index_ == map_->buckets_.size() &&
                other.bucket_index_ == other.map_->buckets_.size()) {
                return true;
            }
            if (map_ != other.map_) return false;
            if (bucket_index_ != other.bucket_index_) return false;
            return bucket_iterator_ == other.bucket_iterator_;
        }

        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }
    };

    class const_iterator {
    private:
        const UnorderedMap* map_;
        std::size_t bucket_index_;
        typename std::list<KeyValuePair>::const_iterator bucket_iterator_;
        friend class UnorderedMap;

    public:
        const_iterator() : map_(nullptr), bucket_index_(0) {}

        const_iterator(const UnorderedMap* m, std::size_t bi,
                       typename std::list<KeyValuePair>::const_iterator it)
            : map_(m), bucket_index_(bi), bucket_iterator_(it) {}

        explicit const_iterator(const iterator& it)
            : map_(it.map_), bucket_index_(it.bucket_index_),
              bucket_iterator_(it.bucket_iterator_) {}

        std::pair<const Key&, const Value&> operator*() const {
            return {bucket_iterator_->key, bucket_iterator_->value};
        }

        PairProxy<const Key&, const Value&> operator->() const {
            return PairProxy<const Key&, const Value&>(
                bucket_iterator_->key, bucket_iterator_->value);
        }

        const_iterator& operator++() {
            ++bucket_iterator_;
            if (bucket_index_ < map_->buckets_.size() &&
                bucket_iterator_ == map_->buckets_[bucket_index_].end()) {
                do {
                    ++bucket_index_;
                } while (bucket_index_ < map_->buckets_.size() &&
                         map_->buckets_[bucket_index_].empty());
                if (bucket_index_ < map_->buckets_.size()) {
                    bucket_iterator_ = map_->buckets_[bucket_index_].begin();
                }
            }
            return *this;
        }

        const_iterator operator++(int) {
            const_iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const const_iterator& other) const {
            if (bucket_index_ == map_->buckets_.size() &&
                other.bucket_index_ == other.map_->buckets_.size()) {
                return true;
            }
            if (map_ != other.map_) return false;
            if (bucket_index_ != other.bucket_index_) return false;
            return bucket_iterator_ == other.bucket_iterator_;
        }

        bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }
    };
};
