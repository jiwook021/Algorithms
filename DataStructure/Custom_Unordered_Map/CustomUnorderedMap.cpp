/**
 * @file CustomUnorderedMap.cpp
 * @brief Implementation of UnorderedMap.
 */

#include "CustomUnorderedMap.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

template <typename Key, typename Value, typename Hash>
std::size_t UnorderedMap<Key, Value, Hash>::GetBucketIndex(
        const Key& key) const {
    return hasher_(key) % buckets_.size();
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::FindInBucket(
        std::list<KeyValuePair>& bucket, const Key& key)
    -> typename std::list<KeyValuePair>::iterator {
    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
        if (it->key == key) return it;
    }
    return bucket.end();
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::FindInBucket(
        const std::list<KeyValuePair>& bucket, const Key& key) const
    -> typename std::list<KeyValuePair>::const_iterator {
    for (auto it = bucket.begin(); it != bucket.end(); ++it) {
        if (it->key == key) return it;
    }
    return bucket.end();
}

template <typename Key, typename Value, typename Hash>
void UnorderedMap<Key, Value, Hash>::RehashImpl(
        std::size_t new_bucket_count) {
    std::vector<std::list<KeyValuePair>> new_buckets(new_bucket_count);
    for (const auto& bucket : buckets_) {
        for (const auto& pair : bucket) {
            std::size_t new_index = hasher_(pair.key) % new_bucket_count;
            new_buckets[new_index].push_back(pair);
        }
    }
    buckets_.swap(new_buckets);
}

// ---------------------------------------------------------------------------
// Construction / assignment
// ---------------------------------------------------------------------------

template <typename Key, typename Value, typename Hash>
UnorderedMap<Key, Value, Hash>::UnorderedMap(
        std::size_t initial_buckets, double max_load_factor)
    : buckets_(initial_buckets > 0 ? initial_buckets : 1),
      element_count_(0),
      max_load_factor_(max_load_factor > 0.0 ? max_load_factor : 1.0) {}

template <typename Key, typename Value, typename Hash>
UnorderedMap<Key, Value, Hash>::UnorderedMap(
        std::initializer_list<std::pair<Key, Value>> init,
        std::size_t initial_buckets, double max_load_factor)
    : buckets_(initial_buckets > 0 ? initial_buckets : 1),
      element_count_(0),
      max_load_factor_(max_load_factor > 0.0 ? max_load_factor : 1.0) {
    for (const auto& pair : init) {
        Insert(pair.first, pair.second);
    }
}

template <typename Key, typename Value, typename Hash>
UnorderedMap<Key, Value, Hash>::UnorderedMap(const UnorderedMap& other)
    : buckets_(other.buckets_), hasher_(other.hasher_),
      element_count_(other.element_count_),
      max_load_factor_(other.max_load_factor_) {}

template <typename Key, typename Value, typename Hash>
UnorderedMap<Key, Value, Hash>::UnorderedMap(UnorderedMap&& other) noexcept
    : buckets_(std::move(other.buckets_)), hasher_(std::move(other.hasher_)),
      element_count_(other.element_count_),
      max_load_factor_(other.max_load_factor_) {
    other.element_count_ = 0;
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::operator=(const UnorderedMap& other)
    -> UnorderedMap& {
    if (this != &other) {
        buckets_ = other.buckets_;
        hasher_ = other.hasher_;
        element_count_ = other.element_count_;
        max_load_factor_ = other.max_load_factor_;
    }
    return *this;
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::operator=(UnorderedMap&& other) noexcept
    -> UnorderedMap& {
    if (this != &other) {
        buckets_ = std::move(other.buckets_);
        hasher_ = std::move(other.hasher_);
        element_count_ = other.element_count_;
        max_load_factor_ = other.max_load_factor_;
        other.element_count_ = 0;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::Insert(const Key& key, const Value& value)
    -> std::pair<iterator, bool> {
    if (LoadFactor() > max_load_factor_) {
        RehashImpl(buckets_.size() * 2);
    }

    std::size_t index = GetBucketIndex(key);
    auto& bucket = buckets_[index];

    auto it = FindInBucket(bucket, key);
    if (it != bucket.end()) {
        return {iterator(this, index, it), false};
    }

    bucket.push_back(KeyValuePair(key, value));
    ++element_count_;
    return {iterator(this, index, std::prev(bucket.end())), true};
}

template <typename Key, typename Value, typename Hash>
Value* UnorderedMap<Key, Value, Hash>::Find(const Key& key) {
    std::size_t index = GetBucketIndex(key);
    auto& bucket = buckets_[index];
    auto it = FindInBucket(bucket, key);
    return (it != bucket.end()) ? &(it->value) : nullptr;
}

template <typename Key, typename Value, typename Hash>
const Value* UnorderedMap<Key, Value, Hash>::Find(const Key& key) const {
    std::size_t index = GetBucketIndex(key);
    const auto& bucket = buckets_[index];
    auto it = FindInBucket(bucket, key);
    return (it != bucket.end()) ? &(it->value) : nullptr;
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::FindIterator(const Key& key)
    -> iterator {
    std::size_t index = GetBucketIndex(key);
    auto& bucket = buckets_[index];
    auto it = FindInBucket(bucket, key);
    return (it != bucket.end()) ? iterator(this, index, it) : end();
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::FindIterator(const Key& key) const
    -> const_iterator {
    std::size_t index = GetBucketIndex(key);
    const auto& bucket = buckets_[index];
    auto it = FindInBucket(bucket, key);
    return (it != bucket.end()) ? const_iterator(this, index, it) : end();
}

template <typename Key, typename Value, typename Hash>
bool UnorderedMap<Key, Value, Hash>::Contains(const Key& key) const {
    std::size_t index = GetBucketIndex(key);
    const auto& bucket = buckets_[index];
    return FindInBucket(bucket, key) != bucket.end();
}

template <typename Key, typename Value, typename Hash>
bool UnorderedMap<Key, Value, Hash>::Erase(const Key& key) {
    std::size_t index = GetBucketIndex(key);
    auto& bucket = buckets_[index];
    auto it = FindInBucket(bucket, key);
    if (it != bucket.end()) {
        bucket.erase(it);
        --element_count_;
        return true;
    }
    return false;
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::Erase(const iterator& pos)
    -> iterator {
    if (pos == end()) return end();

    std::size_t bucket_index = pos.bucket_index_;
    auto bucket_it = pos.bucket_iterator_;
    auto next_it = std::next(bucket_it);

    buckets_[bucket_index].erase(bucket_it);
    --element_count_;

    while (bucket_index < buckets_.size()) {
        if (bucket_index == pos.bucket_index_) {
            if (next_it != buckets_[bucket_index].end()) {
                return iterator(this, bucket_index, next_it);
            }
        } else if (!buckets_[bucket_index].empty()) {
            return iterator(this, bucket_index,
                            buckets_[bucket_index].begin());
        }
        ++bucket_index;
    }
    return end();
}

// ---------------------------------------------------------------------------
// Capacity / bucket queries
// ---------------------------------------------------------------------------

template <typename Key, typename Value, typename Hash>
std::size_t UnorderedMap<Key, Value, Hash>::Size() const {
    return element_count_;
}

template <typename Key, typename Value, typename Hash>
bool UnorderedMap<Key, Value, Hash>::Empty() const {
    return element_count_ == 0;
}

template <typename Key, typename Value, typename Hash>
std::size_t UnorderedMap<Key, Value, Hash>::BucketCount() const {
    return buckets_.size();
}

template <typename Key, typename Value, typename Hash>
std::size_t UnorderedMap<Key, Value, Hash>::BucketSize(std::size_t n) const {
    if (n >= buckets_.size()) {
        throw std::out_of_range("Bucket index out of range");
    }
    return buckets_[n].size();
}

template <typename Key, typename Value, typename Hash>
std::size_t UnorderedMap<Key, Value, Hash>::Bucket(const Key& key) const {
    return GetBucketIndex(key);
}

template <typename Key, typename Value, typename Hash>
double UnorderedMap<Key, Value, Hash>::LoadFactor() const {
    return static_cast<double>(element_count_) / buckets_.size();
}

template <typename Key, typename Value, typename Hash>
double UnorderedMap<Key, Value, Hash>::MaxLoadFactor() const {
    return max_load_factor_;
}

template <typename Key, typename Value, typename Hash>
void UnorderedMap<Key, Value, Hash>::SetMaxLoadFactor(double ml) {
    if (ml <= 0.0) {
        throw std::invalid_argument("Max load factor must be positive");
    }
    max_load_factor_ = ml;
    if (LoadFactor() > max_load_factor_) {
        RehashImpl(static_cast<std::size_t>(
            std::ceil(element_count_ / max_load_factor_)));
    }
}

template <typename Key, typename Value, typename Hash>
void UnorderedMap<Key, Value, Hash>::Rehash(std::size_t count) {
    std::size_t min_size = static_cast<std::size_t>(
        std::ceil(element_count_ / max_load_factor_));
    if (count < min_size) count = min_size;
    if (count > 0) RehashImpl(count);
}

template <typename Key, typename Value, typename Hash>
void UnorderedMap<Key, Value, Hash>::Reserve(std::size_t count) {
    Rehash(static_cast<std::size_t>(
        std::ceil(count / max_load_factor_)));
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

template <typename Key, typename Value, typename Hash>
Value& UnorderedMap<Key, Value, Hash>::operator[](const Key& key) {
    auto it = FindIterator(key);
    if (it == end()) {
        auto [new_it, _] = Insert(key, Value());
        return (*new_it).second;
    }
    return (*it).second;
}

template <typename Key, typename Value, typename Hash>
Value& UnorderedMap<Key, Value, Hash>::At(const Key& key) {
    auto* val = Find(key);
    if (!val) throw std::out_of_range("Key not found in unordered_map");
    return *val;
}

template <typename Key, typename Value, typename Hash>
const Value& UnorderedMap<Key, Value, Hash>::At(const Key& key) const {
    const auto* val = Find(key);
    if (!val) throw std::out_of_range("Key not found in unordered_map");
    return *val;
}

template <typename Key, typename Value, typename Hash>
void UnorderedMap<Key, Value, Hash>::Clear() {
    for (auto& bucket : buckets_) bucket.clear();
    element_count_ = 0;
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::begin() -> iterator {
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
        if (!buckets_[i].empty()) {
            return iterator(this, i, buckets_[i].begin());
        }
    }
    return end();
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::end() -> iterator {
    return iterator(this, buckets_.size(),
                    typename std::list<KeyValuePair>::iterator());
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::begin() const -> const_iterator {
    for (std::size_t i = 0; i < buckets_.size(); ++i) {
        if (!buckets_[i].empty()) {
            return const_iterator(this, i, buckets_[i].begin());
        }
    }
    return end();
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::end() const -> const_iterator {
    return const_iterator(this, buckets_.size(),
        typename std::list<KeyValuePair>::const_iterator());
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::cbegin() const -> const_iterator {
    return begin();
}

template <typename Key, typename Value, typename Hash>
auto UnorderedMap<Key, Value, Hash>::cend() const -> const_iterator {
    return end();
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template class UnorderedMap<int, int>;
template class UnorderedMap<int, std::string>;
template class UnorderedMap<std::string, int>;
template class UnorderedMap<std::string, std::string>;
