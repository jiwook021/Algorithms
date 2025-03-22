# Code Overview: main.cpp

This code implements a **thread-safe hash map** in C++, which is a data structure that allows multiple threads to safely access and modify a collection of key-value pairs concurrently. Let's break down the purpose, functionality, and structure of the code in detail.

---

### **Purpose of the Code**
The primary purpose of this code is to provide a **thread-safe implementation of an unordered map (hash map)**. A hash map is a data structure that stores key-value pairs and allows for fast insertion, lookup, and deletion of elements. However, in a multi-threaded environment, standard hash maps are not safe to use because simultaneous access by multiple threads can lead to race conditions, data corruption, or undefined behavior.

This implementation solves this problem by:
1. Using **fine-grained locking** to allow multiple threads to read from the map simultaneously while ensuring exclusive access for write operations.
2. Providing **thread-safe operations** for inserting, finding, and erasing key-value pairs.
3. Supporting **rehashing** (resizing the hash map) when the number of elements exceeds a specified load factor, ensuring efficient performance even as the map grows.

---

### **Main Functionality**
The code defines a class `ThreadSafeHashMap` that provides the following key features:
1. **Thread-safe operations**: All operations (insert, find, erase) are protected by locks to ensure safe concurrent access.
2. **Fine-grained locking**: Instead of locking the entire map, the code uses **bucket-level locks**, where each bucket (a list of key-value pairs) has its own read-write lock. This allows multiple threads to read from different buckets simultaneously.
3. **Rehashing**: When the map becomes too full (exceeds the maximum load factor), it automatically resizes itself by doubling the number of buckets and redistributing the elements.
4. **Atomic counter**: The number of elements in the map is tracked using an atomic variable (`element_count`) to avoid race conditions during updates.

---

### **Algorithms and Data Structures Used**
1. **Hash Map**:
   - The hash map is implemented using a **vector of buckets**, where each bucket is a **linked list** (`std::list`) of key-value pairs.
   - The hash function (`std::hash<K>`) is used to map keys to bucket indices.

2. **Concurrency Control**:
   - **Read-Write Locks**: Each bucket has a `std::shared_mutex`, which allows multiple threads to acquire a shared (read) lock simultaneously but requires exclusive (write) access for modifications.
   - **Atomic Counter**: The `element_count` variable is atomic, ensuring that updates to the element count are thread-safe.

3. **Rehashing**:
   - When the load factor (number of elements / number of buckets) exceeds the specified threshold, the map is resized by doubling the number of buckets and redistributing the elements.

---

### **Overall Structure**
The code is organized into the following components:

1. **Class Definition**:
   - The `ThreadSafeHashMap` class is a template class that takes three template parameters:
     - `K`: The type of the key.
     - `V`: The type of the value.
     - `Hash`: The hash function to use (defaults to `std::hash<K>`).

2. **Private Members**:
   - `num_buckets`: The number of buckets in the hash map.
   - `buckets`: A vector of linked lists, where each list stores key-value pairs for a specific bucket.
   - `bucket_mutexes`: A vector of read-write locks, one for each bucket.
   - `hash_func`: The hash function used to map keys to buckets.
   - `element_count`: An atomic counter to track the number of elements in the map.
   - `max_load_factor`: The threshold for rehashing.

3. **Private Methods**:
   - `get_bucket_index`: Computes the bucket index for a given key using the hash function.
   - `rehash`: Resizes the map by doubling the number of buckets and redistributing the elements.

4. **Public Methods**:
   - The class provides thread-safe methods for inserting, finding, and erasing key-value pairs (not shown in the provided code snippet but implied by the comments).

5. **Main Function**:
   - The `main` function tests the implementation by:
     - Running basic operations tests.
     - Testing thread safety.
     - Benchmarking the map's performance with different numbers of threads.

---

### **How the Parts Work Together**
1. **Initialization**:
   - When a `ThreadSafeHashMap` object is created, it initializes the number of buckets, the vector of buckets, and the vector of read-write locks.

2. **Insertion**:
   - When a key-value pair is inserted, the code:
     - Computes the bucket index using the hash function.
     - Acquires a write lock for the bucket.
     - Inserts the pair into the bucket's linked list.
     - Updates the atomic element count.
     - Checks if rehashing is needed (if the load factor exceeds the threshold).

3. **Lookup**:
   - When looking up a value by key, the code:
     - Computes the bucket index.
     - Acquires a read lock for the bucket.
     - Searches the bucket's linked list for the key.

4. **Rehashing**:
   - When the map becomes too full, the `rehash` function is called:
     - It locks all bucket mutexes to ensure exclusive access.
     - Doubles the number of buckets and redistributes the elements.
     - Updates the vector of buckets and mutexes.

5. **Concurrency**:
   - Multiple threads can safely access the map because:
     - Read operations use shared locks, allowing concurrent reads.
     - Write operations use exclusive locks, ensuring no two threads modify the same bucket simultaneously.
     - The atomic counter ensures thread-safe updates to the element count.

---

### **Problem Being Solved**
The code solves the problem of **concurrent access to a hash map** in a multi-threaded environment. Without proper synchronization, simultaneous access by multiple threads can lead to race conditions, data corruption, or crashes. This implementation ensures that:
- Multiple threads can safely read from the map simultaneously.
- Write operations are exclusive and thread-safe.
- The map can dynamically resize itself to maintain efficient performance.

---

### **Approach Taken**
The approach taken in this code is to:
1. Use **fine-grained locking** to minimize contention between threads.
2. Leverage **read-write locks** to allow concurrent reads but exclusive writes.
3. Use **atomic variables** to safely track the number of elements.
4. Implement **rehashing** to maintain efficient performance as the map grows.

This approach balances performance and thread safety, making the hash map suitable for use in multi-threaded applications.

---

### **Summary**
This code provides a **thread-safe hash map** that allows multiple threads to safely and efficiently access and modify a collection of key-value pairs. It uses fine-grained locking, read-write locks, and atomic variables to ensure thread safety while maintaining efficient performance. The map also supports dynamic resizing (rehashing) to handle growing numbers of elements. This implementation is well-suited for use in multi-threaded applications where concurrent access to a shared data structure is required.