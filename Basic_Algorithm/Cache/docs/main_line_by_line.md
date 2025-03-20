# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in detail, and provide examples and diagrams where necessary. This explanation assumes no prior knowledge, so I’ll define technical terms and explain concepts as we go.

---

### **1. Header Files**
```cpp
#include <iostream>
#include <unordered_map>
#include <list>
```
- **What it does**: These lines include necessary libraries for the program.
  - `<iostream>`: Provides input/output functionality (e.g., `std::cout` for printing to the console).
  - `<unordered_map>`: Provides a hash map data structure for fast lookups.
  - `<list>`: Provides a doubly linked list data structure for maintaining item order.
- **Why it’s used**: These libraries are essential for implementing the LRU Cache efficiently.

---

### **2. Template Class Definition**
```cpp
template<typename K, typename V>
class LRUCache {
```
- **What it does**: Defines a **template class** called `LRUCache`. 
  - `K` and `V` are **template parameters** representing the key and value types, respectively.
  - This allows the cache to work with any data types (e.g., `std::string` keys and `int` values).
- **Why it’s used**: Templates make the class reusable for different key-value types without rewriting the code.

---

### **3. Private Members**
```cpp
private:
    size_t m_capacity;
    std::list<std::pair<K, V>> m_items;
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> m_cache;
```
- **What it does**: Declares private member variables:
  1. `m_capacity`: Stores the maximum number of items the cache can hold.
  2. `m_items`: A **doubly linked list** that stores key-value pairs in order of usage.
     - The most recently used item is at the **front**, and the least recently used is at the **back**.
  3. `m_cache`: A **hash map** that maps keys to iterators (pointers) in the `m_items` list.
     - This allows O(1) lookups to find items in the list.
- **Why it’s used**:
  - `m_capacity` ensures the cache doesn’t exceed its size limit.
  - `m_items` maintains the order of items for LRU eviction.
  - `m_cache` provides fast access to items in the list.

---

### **4. Constructor**
```cpp
LRUCache(size_t capacity) : m_capacity(capacity) {}
```
- **What it does**: Initializes the cache with a given capacity.
  - `m_capacity` is set to the value passed to the constructor.
- **Why it’s used**: Ensures the cache starts with a defined size limit.

---

### **5. `get` Function**
```cpp
V get(const K& key) {
    if (m_cache.find(key) == m_cache.end()) {
        throw std::out_of_range("Key not found in cache");
    }
    auto item = *m_cache[key];
    m_items.erase(m_cache[key]);
    m_items.push_front(item);
    m_cache[key] = m_items.begin();
    return item.second;
}
```
- **What it does**: Retrieves a value from the cache by its key.
  1. Checks if the key exists in the cache using `m_cache.find(key)`.
     - If the key doesn’t exist, it throws an exception.
  2. If the key exists:
     - Retrieves the item from the list using the iterator stored in `m_cache`.
     - Removes the item from its current position in the list using `m_items.erase`.
     - Moves the item to the front of the list using `m_items.push_front`.
     - Updates the iterator in `m_cache` to point to the new position.
     - Returns the value (`item.second`).
- **Why it’s used**:
  - Ensures that accessing an item marks it as recently used.
  - Maintains the LRU order by moving the accessed item to the front.

---

### **6. `put` Function**
```cpp
void put(const K& key, const V& value) {
    if (m_cache.find(key) != m_cache.end()) {
        m_items.erase(m_cache[key]);
        m_cache.erase(key);
    }
    else if (m_cache.size() >= m_capacity) {
        auto last = m_items.back();
        m_cache.erase(last.first);
        m_items.pop_back();
    }
    m_items.push_front(std::make_pair(key, value));
    m_cache[key] = m_items.begin();
}
```
- **What it does**: Adds a key-value pair to the cache.
  1. If the key already exists:
     - Removes the old item from the list and map.
  2. If the cache is full:
     - Removes the least recently used item (at the back of the list).
  3. Adds the new item to the front of the list and updates the map.
- **Why it’s used**:
  - Ensures the cache doesn’t exceed its capacity.
  - Maintains the LRU order by evicting the least recently used item when necessary.

---

### **7. Utility Functions**
```cpp
bool exists(const K& key) const {
    return m_cache.find(key) != m_cache.end();
}

size_t size() const {
    return m_cache.size();
}

void display() const {
    std::cout << "Cache contents (most recent first):" << std::endl;
    for (const auto& item : m_items) {
        std::cout << item.first << ": " << item.second << std::endl;
    }
    std::cout << "-------------------" << std::endl;
}
```
- **What it does**:
  - `exists`: Checks if a key exists in the cache.
  - `size`: Returns the current number of items in the cache.
  - `display`: Prints the contents of the cache for debugging.
- **Why it’s used**:
  - Provides additional functionality for checking and debugging the cache.

---

### **8. `main` Function**
```cpp
int main() {
    LRUCache<std::string, int> cache(3);
    cache.put("one", 1);
    cache.put("two", 2);
    cache.put("three", 3);
    cache.display();
    std::cout << "Accessing 'one': " << cache.get("one") << std::endl;
    cache.display();
    cache.put("four", 4);
    cache.display();
    try {
        cache.get("two");
    } catch (const std::out_of_range& e) {
        std::cout << "Expected error: " << e.what() << std::endl;
    }
    return 0;
}
```
- **What it does**: Demonstrates the LRU Cache in action.
  1. Creates a cache with a capacity of 3.
  2. Adds three items to the cache.
  3. Accesses an item, moving it to the front.
  4. Adds a fourth item, causing the least recently used item to be evicted.
  5. Attempts to access a non-existent key, resulting in an exception.
- **Why it’s used**:
  - Shows how the cache behaves in real-world scenarios.

---

### **Diagram of the LRU Cache**
Here’s a simple diagram to visualize the cache structure:

```
m_cache (unordered_map):
+--------+-------------------+
| Key    | Iterator in List  |
+--------+-------------------+
| "one"  | -> ["one", 1]     |
| "two"  | -> ["two", 2]     |
| "three"| -> ["three", 3]   |
+--------+-------------------+

m_items (list):
Front -> ["three", 3] -> ["two", 2] -> ["one", 1] <- Back
```

When an item is accessed or added, it moves to the front of the list. When the cache is full, the item at the back is removed.

---

This concludes the detailed explanation of the code. In the next question, we’ll discuss **potential improvements** to make the code even better!