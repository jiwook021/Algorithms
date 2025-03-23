# Code Overview: main.cpp

### Purpose of the Code

This C++ code defines a custom implementation of an **UnorderedMap**, which is essentially a hash table. The purpose of this code is to create a data structure that allows for efficient storage, retrieval, and manipulation of key-value pairs. The UnorderedMap is designed to be generic, meaning it can work with any key and value types, and it uses **hashing** to achieve fast access times.

The code is a simplified version of the `std::unordered_map` found in the C++ Standard Library. It provides similar functionality, such as inserting, erasing, and accessing elements, but with a custom implementation that uses **std::vector** and **std::list** internally to manage the hash table's buckets.

---

### Main Functionality

The UnorderedMap class template provides the following key functionalities:

1. **Storage of Key-Value Pairs**:
   - The map stores key-value pairs (`std::pair<const Key, T>`) in a hash table structure.
   - Keys are unique, meaning each key can only map to one value.

2. **Hashing**:
   - The map uses a hash function (`Hash`) to compute an index (bucket) for each key.
   - This allows for fast access to elements, as the hash function distributes keys evenly across the buckets.

3. **Collision Handling**:
   - Collisions (when two keys hash to the same bucket) are handled using **chaining**, where each bucket is a **std::list** of key-value pairs.
   - If multiple keys hash to the same bucket, they are stored in a linked list within that bucket.

4. **Iteration**:
   - The map provides an iterator to traverse all key-value pairs in the map.
   - The iterator skips empty buckets, ensuring efficient traversal.

5. **Dynamic Resizing**:
   - The map starts with a default number of buckets (`default_bucket_count = 8`).
   - If the number of elements grows too large, the map could be resized to maintain performance (though this functionality is not fully implemented in the provided code).

6. **Common Map Operations**:
   - Insertion (`insert`), access (`at` and `operator[]`), erasure (`erase`), and clearing (`clear`) of elements.
   - Querying the size (`size`), checking if the map is empty (`empty`), and getting the maximum possible size (`max_size`).

---

### Algorithms Used

1. **Hashing**:
   - The hash function (`Hash`) is used to compute the bucket index for a given key.
   - The default hash function is `std::hash<Key>`, which is a standard hashing algorithm provided by the C++ Standard Library.

2. **Chaining for Collision Resolution**:
   - When two keys hash to the same bucket, they are stored in a linked list (`std::list`) within that bucket.
   - This ensures that all keys can be stored and retrieved correctly, even if they collide.

3. **Iterator Traversal**:
   - The iterator skips empty buckets to ensure efficient traversal of the map.
   - It uses a combination of bucket index and list iterator to navigate through the map.

---

### Overall Structure

The code is structured into several parts:

1. **Template Declaration**:
   - The `UnorderedMap` class is a template that takes four parameters:
     - `Key`: The type of the keys.
     - `T`: The type of the values.
     - `Hash`: The hash function type (defaults to `std::hash<Key>`).
     - `KeyEqual`: The key comparison function type (defaults to `std::equal_to<Key>`).

2. **Private Members**:
   - `buckets`: A `std::vector` of `std::list` objects, where each list stores key-value pairs for a specific bucket.
   - `num_elements`: The total number of key-value pairs stored in the map.
   - `hash_func`: The hash function object.
   - `key_equal_obj`: The key comparison function object.
   - `bucket_index`: A helper function that computes the bucket index for a given key.

3. **Public Members**:
   - Constructors and destructor.
   - Capacity-related functions (`empty`, `size`, `max_size`).
   - Iterator implementation.
   - Key operations (`insert`, `erase`, `clear`, `at`, `operator[]`).

4. **Iterator Class**:
   - A nested class that provides iteration over the map.
   - It skips empty buckets and traverses the linked lists within non-empty buckets.

5. **Main Function**:
   - Demonstrates the usage of the `UnorderedMap` class.
   - Inserts key-value pairs, accesses elements, iterates through the map, erases elements, and clears the map.

---

### Problem Being Solved

The problem being solved is the efficient storage and retrieval of key-value pairs. Hash tables are ideal for this purpose because they provide average-case **O(1)** time complexity for insertion, deletion, and lookup operations. The custom implementation in this code demonstrates how a hash table works under the hood, including:

- How keys are hashed to determine their storage location.
- How collisions are handled using chaining.
- How iterators are implemented to traverse the map.

---

### Approach Taken

The approach taken in this code is to use a **hash table with chaining** to implement the map. Here's how the different parts of the code work together:

1. **Hashing**:
   - When a key is inserted, the hash function computes its bucket index.
   - The key-value pair is then added to the corresponding bucket (a `std::list`).

2. **Collision Handling**:
   - If two keys hash to the same bucket, they are stored in the same list.
   - The `key_equal_obj` is used to compare keys within the same bucket.

3. **Iteration**:
   - The iterator starts at the first bucket and skips empty buckets.
   - It traverses the linked lists within non-empty buckets to access all key-value pairs.

4. **Dynamic Resizing**:
   - Although not fully implemented, the map could be resized to maintain performance as the number of elements grows.
   - Resizing would involve rehashing all keys and redistributing them across a larger number of buckets.

---

### How the Parts Work Together

- The **hash function** determines where a key-value pair is stored.
- The **buckets** (a `std::vector` of `std::list` objects) store the key-value pairs.
- The **iterator** provides a way to traverse all key-value pairs in the map.
- The **public member functions** (e.g., `insert`, `erase`, `at`) allow users to interact with the map.

---

### Summary

This code implements a custom hash table (`UnorderedMap`) that provides efficient storage and retrieval of key-value pairs. It uses hashing to distribute keys across buckets and chaining to handle collisions. The implementation is generic, allowing it to work with any key and value types, and it provides common map operations such as insertion, erasure, and iteration. The code demonstrates the core concepts of hash tables and serves as an educational example of how they work.