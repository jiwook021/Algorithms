#include <iostream>
#include <unordered_map>
#include <list>

template<typename K, typename V>
class LRUCache {
private:
    // Capacity of the cache
    size_t m_capacity;
    
    // List to store key-value pairs in order of usage (most recent at front)
    std::list<std::pair<K, V>> m_items;
    
    // Hash map for O(1) lookups, storing key and iterator to list
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> m_cache;

public:
    LRUCache(size_t capacity) : m_capacity(capacity) {}
    
    // Get value from cache
    V get(const K& key) {
        // If key doesn't exist in cache
        if (m_cache.find(key) == m_cache.end()) {
            throw std::out_of_range("Key not found in cache");
        }
        
        // Move the accessed item to the front (most recently used)
        auto item = *m_cache[key];
        m_items.erase(m_cache[key]);
        m_items.push_front(item);
        
        // Update the iterator in the map
        m_cache[key] = m_items.begin();
        
        return item.second;
    }
    
    // Put key-value pair into cache
    void put(const K& key, const V& value) {
        // If key already exists, remove it first
        if (m_cache.find(key) != m_cache.end()) {
            m_items.erase(m_cache[key]);
            m_cache.erase(key);
        }
        // If cache is full, remove the least recently used item (at the back)
        else if (m_cache.size() >= m_capacity) {
            auto last = m_items.back();
            m_cache.erase(last.first);
            m_items.pop_back();
        }
        
        // Add new item to front (most recently used)
        m_items.push_front(std::make_pair(key, value));
        m_cache[key] = m_items.begin();
    }
    
    // Check if key exists in cache
    bool exists(const K& key) const {
        return m_cache.find(key) != m_cache.end();
    }
    
    // Get current size of cache
    size_t size() const {
        return m_cache.size();
    }
    
    // Display all items in cache (for debugging)
    void display() const {
        std::cout << "Cache contents (most recent first):" << std::endl;
        for (const auto& item : m_items) {
            std::cout << item.first << ": " << item.second << std::endl;
        }
        std::cout << "-------------------" << std::endl;
    }
};

// Example usage
int main() {
    // Create an LRU cache with capacity 3
    LRUCache<std::string, int> cache(3);
    
    // Add some items
    cache.put("one", 1);
    cache.put("two", 2);
    cache.put("three", 3);
    
    std::cout << "After adding three items:" << std::endl;
    cache.display();
    
    // Access an item (moves it to front)
    std::cout << "Accessing 'one': " << cache.get("one") << std::endl;
    cache.display();
    
    // Add a new item when cache is full (removes least recently used)
    cache.put("four", 4);
    std::cout << "After adding 'four' (should evict 'two'):" << std::endl;
    cache.display();
    
    // Try to get a key that doesn't exist
    try {
        cache.get("two");
    } catch (const std::out_of_range& e) {
        std::cout << "Expected error: " << e.what() << std::endl;
    }
    
    return 0;
}