#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <functional>
#include <stdexcept>
#include <utility>
#include <initializer_list>
#include <cmath> // Added for std::ceil

template <typename Key, typename Value, typename Hash = std::hash<Key>>
class UnorderedMap {
private:
    // Structure to hold key-value pairs
    struct KeyValuePair {
        Key key;
        Value value;
        
        KeyValuePair(const Key& k, const Value& v) : key(k), value(v) {}
    };
    
    // Each bucket is a linked list of key-value pairs
    using Bucket = std::list<KeyValuePair>;
    
    // Vector of buckets (the hash table)
    std::vector<Bucket> buckets;
    
    // Hash function object
    Hash hasher;
    
    // Number of elements in the map
    size_t element_count;
    
    // Maximum load factor before rehashing
    float max_load_factor_value;
    
    // Hash function to find the appropriate bucket for a key
    size_t getBucketIndex(const Key& key) const {
        // Get hash code and map to a bucket
        size_t hash_code = hasher(key);
        return hash_code % buckets.size();
    }
    
    // Find an element in a bucket with the given key
    typename Bucket::iterator findInBucket(Bucket& bucket, const Key& key) {
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (it->key == key) {
                return it;
            }
        }
        return bucket.end();
    }
    
    // Find an element in a bucket with the given key (const version)
    typename Bucket::const_iterator findInBucket(const Bucket& bucket, const Key& key) const {
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (it->key == key) {
                return it;
            }
        }
        return bucket.end();
    }
    
    // Helper method to rehash the map with a new bucket count
    void rehashImpl(size_t new_bucket_count) {
        // Create a new bucket array
        std::vector<Bucket> new_buckets(new_bucket_count);
        
        // Transfer all elements to the new bucket array
        for (const Bucket& bucket : buckets) {
            for (const KeyValuePair& pair : bucket) {
                size_t new_index = hasher(pair.key) % new_bucket_count;
                new_buckets[new_index].push_back(pair);
            }
        }
        
        // Replace old buckets with new ones
        buckets.swap(new_buckets);
    }

public:
    // Forward declarations for iterator types
    class iterator;
    class const_iterator;
    
    // Helper class for arrow operator
    template <typename K, typename V>
    class PairProxy {
    private:
        std::pair<K, V> pair;
    public:
        PairProxy(K k, V v) : pair(k, v) {}
        const std::pair<K, V>* operator->() const { return &pair; }
    };
    
    // Constructor with initial bucket count
    explicit UnorderedMap(size_t bucket_count = 16) 
        : buckets(bucket_count), element_count(0), max_load_factor_value(1.0) {}
    
    // Constructor with initializer list
    UnorderedMap(std::initializer_list<std::pair<Key, Value>> init, size_t bucket_count = 16)
        : buckets(bucket_count), element_count(0), max_load_factor_value(1.0) {
        for (const auto& pair : init) {
            insert(pair.first, pair.second);
        }
    }
    
    // Copy constructor
    UnorderedMap(const UnorderedMap& other)
        : buckets(other.buckets), hasher(other.hasher), 
          element_count(other.element_count), 
          max_load_factor_value(other.max_load_factor_value) {}
    
    // Move constructor
    UnorderedMap(UnorderedMap&& other) noexcept
        : buckets(std::move(other.buckets)), hasher(std::move(other.hasher)),
          element_count(other.element_count), 
          max_load_factor_value(other.max_load_factor_value) {
        other.element_count = 0;
    }
    
    // Copy assignment operator
    UnorderedMap& operator=(const UnorderedMap& other) {
        if (this != &other) {
            buckets = other.buckets;
            hasher = other.hasher;
            element_count = other.element_count;
            max_load_factor_value = other.max_load_factor_value;
        }
        return *this;
    }
    
    // Move assignment operator
    UnorderedMap& operator=(UnorderedMap&& other) noexcept {
        if (this != &other) {
            buckets = std::move(other.buckets);
            hasher = std::move(other.hasher);
            element_count = other.element_count;
            max_load_factor_value = other.max_load_factor_value;
            other.element_count = 0;
        }
        return *this;
    }
    
    // Destructor
    ~UnorderedMap() = default;
    
    // Insert a key-value pair
    std::pair<iterator, bool> insert(const Key& key, const Value& value) {
        // Check if rehashing is needed
        if (load_factor() > max_load_factor_value) {
            rehashImpl(buckets.size() * 2);
        }
        
        // Find the appropriate bucket
        size_t index = getBucketIndex(key);
        Bucket& bucket = buckets[index];
        
        // Check if key already exists
        auto it = findInBucket(bucket, key);
        if (it != bucket.end()) {
            // Key exists, return iterator to the existing element and false
            return {iterator(this, index, it), false};
        }
        
        // Key doesn't exist, insert new element
        bucket.push_back(KeyValuePair(key, value));
        ++element_count;
        
        // Return iterator to the new element and true
        return {iterator(this, index, std::prev(bucket.end())), true};
    }
    
    // Find an element by key
    iterator find(const Key& key) {
        size_t index = getBucketIndex(key);
        Bucket& bucket = buckets[index];
        auto it = findInBucket(bucket, key);
        
        if (it != bucket.end()) {
            return iterator(this, index, it);
        }
        
        return end();
    }
    
    // Find an element by key (const version)
    const_iterator find(const Key& key) const {
        size_t index = getBucketIndex(key);
        const Bucket& bucket = buckets[index];
        auto it = findInBucket(bucket, key);
        
        if (it != bucket.end()) {
            return const_iterator(this, index, it);
        }
        
        return end();
    }
    
    // Access element by key (creates if doesn't exist)
    Value& operator[](const Key& key) {
        // Try to find the key
        auto it = find(key);
        
        // If not found, insert with default value
        if (it == end()) {
            auto [new_it, _] = insert(key, Value());
            return (*new_it).second;
        }
        
        return (*it).second;
    }
    
    // Access element by key with bounds checking
    Value& at(const Key& key) {
        auto it = find(key);
        if (it == end()) {
            throw std::out_of_range("Key not found in unordered_map");
        }
        return (*it).second;
    }
    
    // Access element by key with bounds checking (const version)
    const Value& at(const Key& key) const {
        auto it = find(key);
        if (it == end()) {
            throw std::out_of_range("Key not found in unordered_map");
        }
        return (*it).second;
    }
    
    // Remove element by key
    size_t erase(const Key& key) {
        size_t index = getBucketIndex(key);
        Bucket& bucket = buckets[index];
        
        // Find the element in the bucket
        auto it = findInBucket(bucket, key);
        if (it != bucket.end()) {
            bucket.erase(it);
            --element_count;
            return 1; // One element erased
        }
        
        return 0; // No elements erased
    }
    
    // Remove element by iterator
    iterator erase(const iterator& it) {
        if (it == end()) {
            return end();
        }
        
        size_t bucket_index = it.bucket_index;
        auto bucket_it = it.bucket_iterator;
        auto next_it = std::next(bucket_it);
        
        buckets[bucket_index].erase(bucket_it);
        --element_count;
        
        // Find the next valid element
        while (bucket_index < buckets.size()) {
            if (bucket_index == it.bucket_index) {
                // If we're in the same bucket, use the saved next iterator
                if (next_it != buckets[bucket_index].end()) {
                    return iterator(this, bucket_index, next_it);
                }
            } else if (!buckets[bucket_index].empty()) {
                // Different bucket with elements
                return iterator(this, bucket_index, buckets[bucket_index].begin());
            }
            
            // Move to the next bucket
            ++bucket_index;
        }
        
        return end();
    }
    
    // Remove all elements
    void clear() {
        for (auto& bucket : buckets) {
            bucket.clear();
        }
        element_count = 0;
    }
    
    // Check if key exists
    bool contains(const Key& key) const {
        size_t index = getBucketIndex(key);
        const Bucket& bucket = buckets[index];
        return findInBucket(bucket, key) != bucket.end();
    }
    
    // Get the number of elements
    size_t size() const {
        return element_count;
    }
    
    // Check if the map is empty
    bool empty() const {
        return element_count == 0;
    }
    
    // Get the number of buckets
    size_t bucket_count() const {
        return buckets.size();
    }
    
    // Get the number of elements in a specific bucket
    size_t bucket_size(size_t n) const {
        if (n >= buckets.size()) {
            throw std::out_of_range("Bucket index out of range");
        }
        return buckets[n].size();
    }
    
    // Get the bucket index for a key
    size_t bucket(const Key& key) const {
        return getBucketIndex(key);
    }
    
    // Get the current load factor
    float load_factor() const {
        return static_cast<float>(element_count) / buckets.size();
    }
    
    // Get the maximum load factor
    float max_load_factor() const {
        return max_load_factor_value;
    }
    
    // Set the maximum load factor
    void max_load_factor(float ml) {
        if (ml <= 0.0f) {
            throw std::invalid_argument("Max load factor must be positive");
        }
        max_load_factor_value = ml;
        
        // Rehash if the current load factor exceeds the new maximum
        if (load_factor() > max_load_factor_value) {
            rehashImpl(std::ceil(element_count / max_load_factor_value));
        }
    }
    
    // Manually trigger rehash with a specified bucket count
    void rehash(size_t count) {
        size_t min_size = std::ceil(element_count / max_load_factor_value);
        if (count < min_size) {
            count = min_size;
        }
        
        if (count > 0) {
            rehashImpl(count); // Call the implementation method to avoid recursion
        }
    }
    
    // Reserve space for at least the specified number of elements
    void reserve(size_t count) {
        rehash(std::ceil(count / max_load_factor_value));
    }
    
    // Get iterator to the beginning of the map
    iterator begin() {
        // Find the first non-empty bucket
        for (size_t i = 0; i < buckets.size(); ++i) {
            if (!buckets[i].empty()) {
                return iterator(this, i, buckets[i].begin());
            }
        }
        return end();
    }
    
    // Get iterator to the end of the map
    iterator end() {
        return iterator(this, buckets.size(), typename Bucket::iterator());
    }
    
    // Get const iterator to the beginning of the map
    const_iterator begin() const {
        // Find the first non-empty bucket
        for (size_t i = 0; i < buckets.size(); ++i) {
            if (!buckets[i].empty()) {
                return const_iterator(this, i, buckets[i].begin());
            }
        }
        return end();
    }
    
    // Get const iterator to the end of the map
    const_iterator end() const {
        return const_iterator(this, buckets.size(), typename Bucket::const_iterator());
    }
    
    // Get const iterator to the beginning of the map
    const_iterator cbegin() const {
        return begin();
    }
    
    // Get const iterator to the end of the map
    const_iterator cend() const {
        return end();
    }

    // Iterator implementation
    class iterator {
    private:
        // Changed from const UnorderedMap* to UnorderedMap* to allow non-const access
        UnorderedMap* map;
        
        // Current bucket index
        size_t bucket_index;
        
        // Iterator to the current element within the current bucket
        typename Bucket::iterator bucket_iterator;
        
        // Allow the map to access private members
        friend class UnorderedMap;
        
    public:
        // Iterator traits
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<const Key, Value>;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::forward_iterator_tag;
        
        // Default constructor
        iterator() : map(nullptr), bucket_index(0) {}
        
        // Constructor with map, bucket index, and bucket iterator
        iterator(UnorderedMap* m, size_t bi, typename Bucket::iterator it)
            : map(m), bucket_index(bi), bucket_iterator(it) {}
        
        // Dereference operator
        std::pair<const Key&, Value&> operator*() const {
            return {bucket_iterator->key, bucket_iterator->value};
        }
        
        // Arrow operator - fixed to return a proxy object
        PairProxy<const Key&, Value&> operator->() const {
            return PairProxy<const Key&, Value&>(bucket_iterator->key, bucket_iterator->value);
        }
        
        // Pre-increment operator
        iterator& operator++() {
            // Move to the next element in the current bucket
            ++bucket_iterator;
            
            // If we reached the end of the current bucket, find the next non-empty bucket
            if (bucket_index < map->buckets.size() && 
                bucket_iterator == map->buckets[bucket_index].end()) {
                do {
                    ++bucket_index;
                } while (bucket_index < map->buckets.size() && 
                         map->buckets[bucket_index].empty());
                
                // If we found a non-empty bucket, set the iterator to its beginning
                if (bucket_index < map->buckets.size()) {
                    bucket_iterator = map->buckets[bucket_index].begin();
                }
            }
            
            return *this;
        }
        
        // Post-increment operator
        iterator operator++(int) {
            iterator temp = *this;
            ++(*this);
            return temp;
        }
        
        // Equality operator
        bool operator==(const iterator& other) const {
            if (bucket_index == map->buckets.size() && 
                other.bucket_index == other.map->buckets.size()) {
                return true; // Both are end iterators
            }
            
            if (map != other.map) {
                return false; // Different maps
            }
            
            if (bucket_index != other.bucket_index) {
                return false; // Different buckets
            }
            
            return bucket_iterator == other.bucket_iterator; // Same position in bucket
        }
        
        // Inequality operator
        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }
    };
    
    // Const iterator implementation
    class const_iterator {
    private:
        // Pointer to the parent map
        const UnorderedMap* map;
        
        // Current bucket index
        size_t bucket_index;
        
        // Iterator to the current element within the current bucket
        typename Bucket::const_iterator bucket_iterator;
        
        // Allow the map to access private members
        friend class UnorderedMap;
        
    public:
        // Iterator traits
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<const Key, const Value>;
        using pointer = value_type*;
        using reference = value_type&;
        using iterator_category = std::forward_iterator_tag;
        
        // Default constructor
        const_iterator() : map(nullptr), bucket_index(0) {}
        
        // Constructor with map, bucket index, and bucket iterator
        const_iterator(const UnorderedMap* m, size_t bi, typename Bucket::const_iterator it)
            : map(m), bucket_index(bi), bucket_iterator(it) {}
        
        // Constructor from non-const iterator (added explicit to fix potential ambiguity)
        explicit const_iterator(const iterator& it)
            : map(it.map), bucket_index(it.bucket_index), bucket_iterator(it.bucket_iterator) {}
        
        // Dereference operator
        std::pair<const Key&, const Value&> operator*() const {
            return {bucket_iterator->key, bucket_iterator->value};
        }
        
        // Arrow operator - fixed to return a proxy object
        PairProxy<const Key&, const Value&> operator->() const {
            return PairProxy<const Key&, const Value&>(bucket_iterator->key, bucket_iterator->value);
        }
        
        // Pre-increment operator
        const_iterator& operator++() {
            // Move to the next element in the current bucket
            ++bucket_iterator;
            
            // If we reached the end of the current bucket, find the next non-empty bucket
            if (bucket_index < map->buckets.size() && 
                bucket_iterator == map->buckets[bucket_index].end()) {
                do {
                    ++bucket_index;
                } while (bucket_index < map->buckets.size() && 
                         map->buckets[bucket_index].empty());
                
                // If we found a non-empty bucket, set the iterator to its beginning
                if (bucket_index < map->buckets.size()) {
                    bucket_iterator = map->buckets[bucket_index].begin();
                }
            }
            
            return *this;
        }
        
        // Post-increment operator
        const_iterator operator++(int) {
            const_iterator temp = *this;
            ++(*this);
            return temp;
        }
        
        // Equality operator
        bool operator==(const const_iterator& other) const {
            if (bucket_index == map->buckets.size() && 
                other.bucket_index == other.map->buckets.size()) {
                return true; // Both are end iterators
            }
            
            if (map != other.map) {
                return false; // Different maps
            }
            
            if (bucket_index != other.bucket_index) {
                return false; // Different buckets
            }
            
            return bucket_iterator == other.bucket_iterator; // Same position in bucket
        }
        
        // Inequality operator
        bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }
    };
};


int main() {
    // Create an unordered map with string keys and int values
    UnorderedMap<std::string, int> map;
    
    // Insert some elements
    map.insert("apple", 5);
    map.insert("banana", 8);
    map.insert("orange", 10);
    
    // Access elements using operator[]
    std::cout << "apple: " << map["apple"] << std::endl;
    
    // Use operator[] to add a new element
    map["grape"] = 12;
    
    // Iterate through all elements
    std::cout << "All elements:" << std::endl;
    for (auto it = map.begin(); it != map.end(); ++it) {
        std::cout << (*it).first << ": " << (*it).second << std::endl;
    }
    
    // Check if a key exists
    std::cout << "Contains 'pear': " << (map.contains("pear") ? "yes" : "no") << std::endl;
    
    // Get the number of elements and buckets
    std::cout << "Size: " << map.size() << std::endl;
    std::cout << "Bucket count: " << map.bucket_count() << std::endl;
    
    // Remove an element
    map.erase("banana");
    std::cout << "After removing 'banana', size: " << map.size() << std::endl;
    
    // Clear the map
    map.clear();
    std::cout << "After clearing, size: " << map.size() << std::endl;
    
    return 0;
}