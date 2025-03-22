# Code Overview: main.cpp

This code implements a custom **UnorderedMap** class in C++, which is essentially a hash table implementation. Let's break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The code defines a custom hash table (UnorderedMap) that stores key-value pairs. It is similar to the `std::unordered_map` in the C++ Standard Library, but it is implemented from scratch. The purpose of this code is to provide a data structure that allows for:
1. **Fast insertion, deletion, and lookup** of key-value pairs.
2. **Efficient storage** of elements using a hash table.
3. **Dynamic resizing** of the hash table to maintain performance as the number of elements grows.

The hash table uses **chaining** (linked lists) to handle collisions, meaning that multiple keys that hash to the same bucket are stored in a linked list within that bucket.

---

### **Main Functionality**
The UnorderedMap class provides the following core functionalities:
1. **Insertion**: Add a new key-value pair to the map.
2. **Lookup**: Retrieve the value associated with a given key.
3. **Deletion**: Remove a key-value pair from the map.
4. **Iteration**: Traverse all key-value pairs in the map.
5. **Rehashing**: Dynamically resize the hash table to maintain performance as the number of elements grows.

---

### **Algorithms Used**
1. **Hashing**:
   - The hash table uses a hash function (`std::hash` by default) to map keys to bucket indices.
   - The `getBucketIndex` method computes the bucket index for a given key using the formula:  
     `hash_code % buckets.size()`.

2. **Collision Handling**:
   - Collisions (when two keys hash to the same bucket) are handled using **chaining**.
   - Each bucket is a linked list (`std::list`) of key-value pairs.

3. **Rehashing**:
   - When the load factor (number of elements / number of buckets) exceeds the maximum load factor, the hash table is resized.
   - The `rehashImpl` method creates a new vector of buckets, recalculates the bucket indices for all elements, and transfers them to the new buckets.

4. **Iteration**:
   - The map provides iterators to traverse all key-value pairs in the hash table.

---

### **Overall Structure**
The code is organized into the following components:

1. **Private Members**:
   - `KeyValuePair`: A structure to store key-value pairs.
   - `Bucket`: A type alias for a linked list (`std::list`) of key-value pairs.
   - `buckets`: A vector of buckets (the hash table itself).
   - `hasher`: The hash function object.
   - `element_count`: The number of elements in the map.
   - `max_load_factor_value`: The maximum load factor before rehashing.

2. **Private Methods**:
   - `getBucketIndex`: Computes the bucket index for a given key.
   - `findInBucket`: Searches for a key in a specific bucket.
   - `rehashImpl`: Resizes the hash table and rehashes all elements.

3. **Public Members**:
   - `iterator` and `const_iterator`: Iterator types for traversing the map.
   - `PairProxy`: A helper class to support the arrow operator (`->`) for iterators.

4. **Public Methods**:
   - Constructors: Initialize the map with a default or custom bucket count.
   - `insert`: Adds a key-value pair to the map.
   - `operator[]`: Accesses or inserts a value for a given key.
   - `erase`: Removes a key-value pair from the map.
   - `contains`: Checks if a key exists in the map.
   - `size`: Returns the number of elements in the map.
   - `bucket_count`: Returns the number of buckets in the hash table.

5. **Main Function**:
   - Demonstrates the usage of the UnorderedMap class by inserting, accessing, and iterating over key-value pairs.

---

### **How the Parts Work Together**
1. **Initialization**:
   - The map is initialized with a default or custom number of buckets.
   - The hash function (`std::hash`) is used to compute bucket indices.

2. **Insertion**:
   - When a key-value pair is inserted, the `getBucketIndex` method computes the bucket index.
   - The `findInBucket` method checks if the key already exists in the bucket.
   - If the key exists, its value is updated; otherwise, a new key-value pair is added to the bucket.

3. **Lookup**:
   - The `operator[]` or `findInBucket` method is used to retrieve the value for a given key.

4. **Rehashing**:
   - When the load factor exceeds the maximum load factor, the `rehashImpl` method is called to resize the hash table and redistribute the elements.

5. **Iteration**:
   - The `iterator` and `const_iterator` classes allow traversal of all key-value pairs in the map.

---

### **Problem Being Solved**
The code solves the problem of efficiently storing and retrieving key-value pairs using a hash table. Hash tables are ideal for this purpose because they provide average-case O(1) time complexity for insertion, deletion, and lookup operations. The use of chaining ensures that collisions are handled gracefully, and rehashing maintains performance as the number of elements grows.

---

### **Approach Taken**
1. **Hash Table with Chaining**:
   - The hash table is implemented as a vector of linked lists (buckets).
   - Each bucket stores key-value pairs that hash to the same index.

2. **Dynamic Resizing**:
   - The hash table automatically resizes when the load factor exceeds a threshold, ensuring that performance does not degrade as the number of elements increases.

3. **Iterators**:
   - Custom iterator classes are implemented to support traversal of the hash table.

4. **STL Integration**:
   - The code uses standard library components like `std::vector`, `std::list`, and `std::hash` to simplify implementation.

---

### **Summary**
This code implements a custom hash table (UnorderedMap) that provides fast and efficient storage and retrieval of key-value pairs. It uses hashing, chaining, and dynamic resizing to achieve its goals. The structure is modular, with clear separation of concerns between private and public members, and it demonstrates good use of C++ features like templates, iterators, and standard library components.