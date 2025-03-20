# Explain how `std::map`, `std::unordered_map`, and `std::vector` work internally.

### `std::map`

`std::map` in C++ is a sorted associative container that stores elements formed by a combination of a key value and a mapped value, following a specific order. Here's a breakdown of its internal workings:

1. **Underlying Structure**: `std::map` typically utilizes a red-black tree, which is a type of self-balancing binary search tree. Each node of the tree contains the elements (pairs of key and mapped value), and the tree maintains a sorted order by the key. 

2. **Insertion**: When a new element is inserted, the tree performs a binary search to locate the correct position for the new key and inserts the new element there. After insertion, the tree may need to rebalance itself to maintain its properties. The balancing helps in keeping the operations optimal.

3. **Search**: To find an element, the map uses the key to perform a binary search through the tree, which operates in O(log n) time complexity, where n is the number of elements in the map.

4. **Deletion**: Removing an element involves finding the node and then re-adjusting the tree structure to maintain its balanced nature.

5. **Order**: Since the elements are sorted according to the key, the order is deterministic, and the in-order traversal of the tree will yield the elements in their sorted key order.

### `std::unordered_map`

`std::unordered_map` is a hash table based associative container that stores elements formed by the combination of a key value and a mapped value. It provides faster average complexity for access operations but does not order elements. Here’s how it works:

1. **Hashing Mechanism**: Each key is passed through a hash function which converts the key into a hash code. This hash code is then used to find an index in an array where the value will be stored.

2. **Handling Collisions**: Collisions occur when different keys produce the same hash code or are mapped to the same index. `std::unordered_map` can handle collisions using several techniques, one common method being chaining. In chaining, each cell of the hash table points to a linked list of records that have the same hash.

3. **Insertion, Search, and Deletion**: These operations involve computing the hash of the key and accessing the appropriate index in the array. In the absence of many collisions, these operations can approach O(1) time complexity on average, though poor hash functions or high load factors can degrade performance to O(n).

4. **Resizing**: As more elements are inserted, the load factor (ratio of number of elements to the number of buckets) increases, which can increase the likelihood of collisions. `std::unordered_map` may need to resize and rehash all entries periodically to maintain efficiency.

### `std::vector`

`std::vector` is a dynamic array that allows random access to its elements and provides efficient insertion and deletion at its end.

1. **Array-based Structure**: Internally, `std::vector` uses a dynamically allocated array to store its elements. This array may need to be resized and its elements copied to a new larger array when more elements are added than can be accommodated in the current array.

2. **Insertion**: Adding an element to the end of the vector (using `push_back`) is very efficient if the vector has pre-allocated space (capacity). If the current array is full, the vector will allocate a new array typically twice the size of the old one, copy the existing elements to the new array, and then add the new element.

3. **Access**: Accessing elements by index is very fast (O(1)) because it involves direct indexing into the underlying array.

4. **Deletion**: Removing elements from the end of the vector (`pop_back`) is efficient. However, removing elements from the middle of the vector is less efficient (O(n)) as it requires shifting all the subsequent elements to fill the gap.

Each of these data structures is designed to offer particular advantages for different use cases, balancing between efficiency of operations like insertion, deletion, search, and memory usage.