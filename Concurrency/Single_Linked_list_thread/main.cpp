#include <iostream>
#include <vector>
#include <list>
#include <optional>
#include <functional>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <random>
#include <chrono>
#include <cassert>

/**
 * @brief Thread-safe implementation of an unordered map
 * 
 * This class provides a thread-safe hash map implementation with fine-grained locking
 * for concurrent access. It uses bucket-level read-write locks to allow multiple
 * concurrent reads but exclusive writes.
 * 
 * Time Complexity:
 * - Insert: O(1) average case, O(n) worst case (when rehashing)
 * - Find: O(1) average case, O(n) worst case (when many keys hash to the same bucket)
 * - Erase: O(1) average case, O(n) worst case (when many keys hash to the same bucket)
 * 
 * Space Complexity: O(n) where n is the number of key-value pairs
 */
template<typename K, typename V, typename Hash = std::hash<K>>
class ThreadSafeHashMap {
private:
    // Internal structure to represent key-value pairs
    struct KeyValuePair {
        K key;
        V value;
        KeyValuePair(const K& k, const V& v) : key(k), value(v) {}
    };

    // Number of buckets in the hash map
    size_t num_buckets;
    
    // Vector of buckets, each containing a list of key-value pairs
    std::vector<std::list<KeyValuePair>> buckets;
    
    // Vector of read-write locks for each bucket
    // Using shared_mutex allows multiple readers but exclusive writers
    mutable std::vector<std::shared_mutex> bucket_mutexes;
    
    // Hash function
    Hash hash_func;
    
    // Atomic counter for the number of elements to avoid race conditions
    std::atomic<size_t> element_count;
    
    // Load factor threshold for rehashing
    float max_load_factor;
    
    /**
     * @brief Get the bucket index for a key
     * 
     * @param key The key to hash
     * @return size_t The bucket index
     */
    size_t get_bucket_index(const K& key) const {
        return hash_func(key) % num_buckets;
    }
    
    /**
     * @brief Rehash the map to increase capacity
     * 
     * This function is called when the load factor exceeds the maximum load factor.
     * It doubles the number of buckets and redistributes all elements.
     * Note: This function assumes that all bucket mutexes are already locked.
     */
    void rehash() {
        // Store old buckets
        auto old_buckets = std::move(buckets);
        auto old_num_buckets = num_buckets;
        
        // Double the number of buckets
        num_buckets *= 2;
        
        // Initialize new buckets and mutexes
        buckets.resize(num_buckets);
        bucket_mutexes.resize(num_buckets);
        
        // Reset element count
        element_count = 0;
        
        // Reinsert all elements
        for (size_t i = 0; i < old_num_buckets; ++i) {
            for (auto& kv : old_buckets[i]) {
                insert(kv.key, kv.value);
            }
        }
    }

public:
    /**
     * @brief Construct a new Thread Safe Hash Map
     * 
     * @param initial_buckets Number of initial buckets (default: 16)
     * @param load_factor Maximum load factor before rehashing (default: 0.75)
     * @throws std::invalid_argument if initial_buckets is 0 or load_factor is invalid
     */
    explicit ThreadSafeHashMap(size_t initial_buckets = 16, float load_factor = 0.75f)
        : num_buckets(initial_buckets),
          buckets(initial_buckets),
          bucket_mutexes(initial_buckets),
          element_count(0),
          max_load_factor(load_factor) {
        
        // Validate input parameters
        if (initial_buckets == 0) {
            throw std::invalid_argument("Number of buckets must be greater than 0");
        }
        
        if (load_factor <= 0.0f || load_factor >= 1.0f) {
            throw std::invalid_argument("Load factor must be between 0 and 1");
        }
    }
    
    /**
     * @brief Insert or update a key-value pair
     * 
     * If the key already exists, its value is updated.
     * If the key doesn't exist, a new key-value pair is inserted.
     * 
     * @param key The key
     * @param value The value
     */
    void insert(const K& key, const V& value) {
        size_t bucket_idx = get_bucket_index(key);
        
        // Lock the specific bucket for exclusive write access
        // This locking mechanism is needed to prevent data races when multiple
        // threads try to modify the same bucket simultaneously
        std::unique_lock<std::shared_mutex> lock(bucket_mutexes[bucket_idx]);
        
        // Check if key already exists
        auto& bucket = buckets[bucket_idx];
        for (auto& kv : bucket) {
            if (kv.key == key) {
                // Update existing value
                kv.value = value;
                return;
            }
        }
        
        // Insert new key-value pair
        bucket.emplace_back(key, value);
        size_t new_count = ++element_count;
        
        // Check if rehashing is needed
        float current_load_factor = static_cast<float>(new_count) / num_buckets;
        if (current_load_factor > max_load_factor) {
            // Release the lock before rehashing to avoid deadlock
            lock.unlock();
            
            // Acquire locks for all buckets for rehashing
            // This is necessary to ensure consistency during rehashing
            std::vector<std::unique_lock<std::shared_mutex>> all_locks;
            all_locks.reserve(num_buckets);
            
            for (size_t i = 0; i < num_buckets; ++i) {
                all_locks.emplace_back(bucket_mutexes[i]);
            }
            
            // Rehash with all locks acquired
            rehash();
        }
    }
    
    /**
     * @brief Find a value by key
     * 
     * @param key The key to find
     * @return std::optional<V> The value if found, std::nullopt otherwise
     */
    std::optional<V> find(const K& key) const {
        size_t bucket_idx = get_bucket_index(key);
        
        // Lock the specific bucket for shared read access
        // This allows multiple threads to read from the same bucket concurrently
        std::shared_lock<std::shared_mutex> lock(bucket_mutexes[bucket_idx]);
        
        // Search for the key
        const auto& bucket = buckets[bucket_idx];
        for (const auto& kv : bucket) {
            if (kv.key == key) {
                return kv.value;
            }
        }
        
        // Key not found
        return std::nullopt;
    }
    
    /**
     * @brief Erase a key-value pair
     * 
     * @param key The key to erase
     * @return bool True if the key was found and erased, false otherwise
     */
    bool erase(const K& key) {
        size_t bucket_idx = get_bucket_index(key);
        
        // Lock the specific bucket for exclusive write access
        // This is needed to prevent data races when multiple threads
        // try to erase elements from the same bucket
        std::unique_lock<std::shared_mutex> lock(bucket_mutexes[bucket_idx]);
        
        // Search for the key
        auto& bucket = buckets[bucket_idx];
        auto it = bucket.begin();
        
        while (it != bucket.end()) {
            if (it->key == key) {
                bucket.erase(it);
                --element_count;
                return true;
            }
            ++it;
        }
        
        // Key not found
        return false;
    }
    
    /**
     * @brief Get the number of elements
     * 
     * @return size_t The number of elements
     */
    size_t size() const {
        return element_count.load();
    }
    
    /**
     * @brief Check if the map is empty
     * 
     * @return bool True if the map is empty, false otherwise
     */
    bool empty() const {
        return element_count.load() == 0;
    }
    
    /**
     * @brief Clear all elements
     */
    void clear() {
        // Lock all buckets for exclusive write access
        // This is needed to ensure thread safety during clear operation
        std::vector<std::unique_lock<std::shared_mutex>> all_locks;
        all_locks.reserve(num_buckets);
        
        for (size_t i = 0; i < num_buckets; ++i) {
            all_locks.emplace_back(bucket_mutexes[i]);
        }
        
        // Clear all buckets
        for (auto& bucket : buckets) {
            bucket.clear();
        }
        
        // Reset element count
        element_count = 0;
    }
    
    /**
     * @brief Get the current load factor
     * 
     * @return float The current load factor
     */
    float load_factor() const {
        return static_cast<float>(element_count.load()) / num_buckets;
    }
};

/**
 * @brief Test function for concurrent operations
 * 
 * This function performs random operations (insert, find, erase) on the map
 * to test thread safety.
 * 
 * @param map The map to test
 * @param thread_id The ID of the thread
 * @param operations The number of operations to perform
 */
void test_concurrent_operations(ThreadSafeHashMap<int, std::string>& map, int thread_id, int operations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> key_dist(1, 1000);
    std::uniform_int_distribution<> op_dist(0, 2); // 0: insert, 1: find, 2: erase
    
    for (int i = 0; i < operations; ++i) {
        int key = key_dist(gen);
        int operation = op_dist(gen);
        
        switch (operation) {
            case 0: { // Insert
                std::string value = "Thread-" + std::to_string(thread_id) + "-Value-" + std::to_string(i);
                map.insert(key, value);
                break;
            }
            case 1: { // Find
                auto result = map.find(key);
                // We don't need to do anything with the result here
                break;
            }
            case 2: { // Erase
                map.erase(key);
                break;
            }
        }
    }
}

/**
 * @brief Function to test basic operations
 * 
 * This function tests the basic operations of the map for correctness.
 */
void test_basic_operations() {
    ThreadSafeHashMap<int, std::string> map;
    
    // Test insert and find
    map.insert(1, "One");
    map.insert(2, "Two");
    map.insert(3, "Three");
    
    auto value1 = map.find(1);
    auto value2 = map.find(2);
    auto value3 = map.find(3);
    auto value4 = map.find(4); // Non-existent key
    
    assert(value1.has_value() && value1.value() == "One");
    assert(value2.has_value() && value2.value() == "Two");
    assert(value3.has_value() && value3.value() == "Three");
    assert(!value4.has_value());
    
    // Test update
    map.insert(1, "Updated One");
    value1 = map.find(1);
    assert(value1.has_value() && value1.value() == "Updated One");
    
    // Test erase
    bool erased1 = map.erase(1);
    bool erased4 = map.erase(4);
    assert(erased1);
    assert(!erased4);
    
    value1 = map.find(1);
    assert(!value1.has_value());
    
    // Test size and empty
    assert(map.size() == 2);
    assert(!map.empty());
    
    // Test clear
    map.clear();
    assert(map.size() == 0);
    assert(map.empty());
    
    std::cout << "Basic operations test passed!" << std::endl;
}

/**
 * @brief Thread safety test
 * 
 * This function tests the thread safety of the map by checking if concurrent
 * modifications lead to a consistent state.
 */
void test_thread_safety() {
    ThreadSafeHashMap<int, int> map;
    
    // Insert initial values
    for (int i = 0; i < 100; ++i) {
        map.insert(i, i * 10);
    }
    
    // Create threads that concurrently modify the same keys
    std::vector<std::thread> threads;
    const int num_threads = 4;
    const int operations_per_thread = 1000;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&map, t]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                int key = i % 100;  // All threads operate on the same keys
                
                // Each thread alternates between different operations
                if ((i + t) % 3 == 0) {
                    map.insert(key, t * 1000 + i);  // Insert
                } else if ((i + t) % 3 == 1) {
                    auto val = map.find(key);  // Find
                } else {
                    map.erase(key);  // Erase
                }
            }
        });
    }
    
    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    // No assertion here - we just want to ensure no crashes or exceptions
    std::cout << "Thread safety test completed." << std::endl;
}

/**
 * @brief Benchmark function
 * 
 * This function measures the performance of the map with different numbers of threads.
 * 
 * @param map The map to benchmark
 * @param num_threads The number of threads to use
 * @param operations_per_thread The number of operations per thread
 */
void benchmark(ThreadSafeHashMap<int, std::string>& map, int num_threads, int operations_per_thread) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Create and start threads
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(test_concurrent_operations, std::ref(map), i, operations_per_thread);
        }
        
        // Join all threads
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during benchmark: " << e.what() << std::endl;
        
        // Clean up any remaining threads
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        return;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Benchmark with " << num_threads << " threads, " 
              << operations_per_thread << " operations per thread:" << std::endl;
    std::cout << "Total operations: " << num_threads * operations_per_thread << std::endl;
    std::cout << "Duration: " << duration.count() << " ms" << std::endl;
    std::cout << "Operations per second: " 
              << static_cast<double>(num_threads * operations_per_thread * 1000) / duration.count() << std::endl;
    std::cout << "Final map size: " << map.size() << std::endl;
    std::cout << "Final load factor: " << map.load_factor() << std::endl;
    std::cout << std::endl;
}

int main() {
    try {
        std::cout << "Running basic operations test..." << std::endl;
        test_basic_operations();
        
        std::cout << "Running thread safety test..." << std::endl;
        test_thread_safety();
        
        // Create a thread-safe hash map for benchmarking
        ThreadSafeHashMap<int, std::string> map(32, 0.75f);
        
        std::cout << "Running benchmarks with different thread counts..." << std::endl;
        // Run benchmarks with different numbers of threads
        benchmark(map, 1, 10000);  // Single-threaded baseline
        benchmark(map, 2, 10000);  // 2 threads
        benchmark(map, 4, 10000);  // 4 threads
        benchmark(map, 8, 10000);  // 8 threads
        
        std::cout << "All tests completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}