# Code Overview: main.cpp

This code implements a **Least Recently Used (LRU) Cache**, a common data structure used in computer systems to optimize memory usage and improve performance. Let’s break down the purpose, functionality, and structure of the code in detail.

---

### **Purpose of the Code**
The purpose of this code is to implement an **LRU Cache**, which is a cache eviction policy that removes the least recently used items when the cache reaches its capacity. This is useful in scenarios where you want to store a limited number of items in memory but still ensure that the most frequently or recently accessed items are readily available.

For example:
- Web browsers use LRU caches to store recently visited web pages.
- Databases use LRU caches to store frequently accessed query results.
- Operating systems use LRU caches to manage memory pages.

The LRU Cache ensures that the most recently accessed items are kept in the cache, while the least recently used items are evicted when the cache is full.

---

### **Main Functionality**
The code implements the following key functionalities:
1. **Cache Initialization**: The cache is initialized with a fixed capacity.
2. **Insertion (`put`)**: Adds a new key-value pair to the cache. If the cache is full, it removes the least recently used item.
3. **Retrieval (`get`)**: Retrieves a value by its key. If the key exists, the item is moved to the front of the cache (marking it as recently used).
4. **Existence Check (`exists`)**: Checks if a key exists in the cache.
5. **Size Check (`size`)**: Returns the current number of items in the cache.
6. **Display (`display`)**: Prints the contents of the cache for debugging purposes.

---

### **Algorithms and Data Structures Used**
The code uses two main data structures to achieve efficient operations:
1. **Doubly Linked List (`std::list`)**: 
   - Used to maintain the order of items based on their usage.
   - The most recently used item is at the **front** of the list.
   - The least recently used item is at the **back** of the list.
   - This allows for efficient reordering of items when they are accessed.

2. **Hash Map (`std::unordered_map`)**:
   - Used to store key-value pairs, where the value is an iterator to the corresponding item in the linked list.
   - Provides **O(1)** average time complexity for lookups, insertions, and deletions.

The combination of these two data structures allows the LRU Cache to achieve **O(1)** time complexity for both `get` and `put` operations.

---

### **How the Code Works Together**
1. **Initialization**:
   - The cache is initialized with a fixed capacity (`m_capacity`).
   - The `m_items` list stores the key-value pairs in order of usage.
   - The `m_cache` map stores the keys and their corresponding iterators in the list.

2. **Insertion (`put`)**:
   - If the key already exists, the old item is removed from the list and map.
   - If the cache is full, the least recently used item (at the back of the list) is removed.
   - The new item is added to the front of the list, and its iterator is stored in the map.

3. **Retrieval (`get`)**:
   - If the key exists, the corresponding item is moved to the front of the list (marking it as recently used).
   - The iterator in the map is updated to point to the new position in the list.
   - If the key doesn’t exist, an exception is thrown.

4. **Existence Check (`exists`)**:
   - Checks if a key exists in the map.

5. **Size Check (`size`)**:
   - Returns the number of items currently in the cache.

6. **Display (`display`)**:
   - Iterates through the list and prints the key-value pairs in order of usage.

---

### **Example Usage**
The `main` function demonstrates how the LRU Cache works:
1. A cache with a capacity of 3 is created.
2. Three items are added to the cache.
3. An item is accessed, moving it to the front of the cache.
4. A fourth item is added, causing the least recently used item to be evicted.
5. An attempt to access a non-existent key results in an exception.

---

### **Key Takeaways**
- The LRU Cache is a powerful tool for optimizing memory usage and improving performance in systems with limited resources.
- The combination of a doubly linked list and a hash map ensures efficient operations with **O(1)** time complexity.
- The code is modular, with clear separation of concerns between cache operations (insertion, retrieval, eviction) and utility functions (existence check, size check, display).

This implementation is a classic example of how to use data structures effectively to solve real-world problems. In the next question, we’ll dive into a **line-by-line explanation** of the code to understand how each part works in detail.