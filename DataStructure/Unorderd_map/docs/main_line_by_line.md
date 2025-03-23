# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll start from the top and work my way down, ensuring that every concept is explained clearly and thoroughly.

---

### **1. Header Includes**
```cpp
#include <iostream>
#include <vector>
#include <list>
#include <functional>
#include <utility>
#include <stdexcept>
#include <iterator>
#include <limits>
#include <algorithm>
```

#### What It Does:
These lines include necessary C++ Standard Library headers that provide functionality for input/output, data structures, and algorithms.

#### Explanation:
- **`<iostream>`**: Provides input/output functionality (e.g., `std::cout` for printing to the console).
- **`<vector>`**: Provides the `std::vector` class, a dynamic array that can grow or shrink in size.
- **`<list>`**: Provides the `std::list` class, a doubly linked list.
- **`<functional>`**: Provides tools for working with functions and function objects (e.g., `std::hash`).
- **`<utility>`**: Provides utilities like `std::pair`, which is used to store key-value pairs.
- **`<stdexcept>`**: Provides standard exception classes (e.g., `std::out_of_range`).
- **`<iterator>`**: Provides tools for working with iterators.
- **`<limits>`**: Provides information about the limits of numeric types (e.g., `std::numeric_limits<size_t>::max()`).
- **`<algorithm>`**: Provides algorithms like sorting and searching.

#### Why These Are Used:
These headers are included because the code relies on standard library components like vectors, lists, and hash functions. They provide the building blocks for the custom `UnorderedMap` implementation.

---

### **2. Template Declaration**
```cpp
template <typename Key, typename T, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class UnorderedMap {
```

#### What It Does:
This declares a **class template** named `UnorderedMap`. It is a generic class that can work with any key type (`Key`), value type (`T`), hash function (`Hash`), and key comparison function (`KeyEqual`).

#### Explanation:
- **Template Parameters**:
  - `Key`: The type of the keys (e.g., `int`, `std::string`).
  - `T`: The type of the values (e.g., `std::string`, `double`).
  - `Hash`: The hash function type (defaults to `std::hash<Key>`).
  - `KeyEqual`: The key comparison function type (defaults to `std::equal_to<Key>`).

#### Why Templates Are Used:
Templates allow the `UnorderedMap` to work with any data type. For example, you can create an `UnorderedMap<int, std::string>` to map integers to strings, or an `UnorderedMap<std::string, double>` to map strings to doubles.

---

### **3. Private Members**
```cpp
private:
    static const size_t default_bucket_count = 8;
    std::vector<std::list<std::pair<const Key, T>>> buckets;
    size_t num_elements;
    Hash hash_func;
    KeyEqual key_equal_obj;
```

#### What It Does:
These are the private members of the `UnorderedMap` class. They store the internal state of the map.

#### Explanation:
- **`default_bucket_count`**: The default number of buckets (initially 8). A **bucket** is a slot in the hash table where key-value pairs are stored.
- **`buckets`**: A `std::vector` of `std::list` objects. Each bucket is a linked list (`std::list`) that stores key-value pairs (`std::pair<const Key, T>`).
- **`num_elements`**: The total number of key-value pairs stored in the map.
- **`hash_func`**: The hash function object (e.g., `std::hash<Key>`).
- **`key_equal_obj`**: The key comparison function object (e.g., `std::equal_to<Key>`).

#### Why These Are Used:
- **`buckets`**: The vector of lists is used to implement **chaining**, a collision resolution technique. If two keys hash to the same bucket, they are stored in the same list.
- **`hash_func`**: Computes the bucket index for a given key.
- **`key_equal_obj`**: Compares keys to check for equality (e.g., when searching for a key in a bucket).

---

### **4. Helper Function: `bucket_index`**
```cpp
size_t bucket_index(const Key &key) const {
    return hash_func(key) % buckets.size();
}
```

#### What It Does:
This function computes the bucket index for a given key.

#### Explanation:
- **`hash_func(key)`**: Calls the hash function to compute a hash value for the key.
- **`% buckets.size()`**: Uses the modulo operator to ensure the hash value fits within the range of bucket indices.

#### Example:
If `buckets.size()` is 8 and `hash_func(key)` returns 123, then `123 % 8 = 3`, so the key is stored in bucket 3.

#### Why This Is Used:
This ensures that keys are distributed evenly across the buckets, which is essential for efficient hash table performance.

---

### **5. Constructor**
```cpp
UnorderedMap()
    : buckets(default_bucket_count),
      num_elements(0),
      hash_func(Hash()),
      key_equal_obj(KeyEqual()) {}
```

#### What It Does:
This is the default constructor for the `UnorderedMap` class. It initializes the map with default values.

#### Explanation:
- **`buckets(default_bucket_count)`**: Initializes the `buckets` vector with 8 empty lists.
- **`num_elements(0)`**: Initializes the element count to 0.
- **`hash_func(Hash())`**: Initializes the hash function object.
- **`key_equal_obj(KeyEqual())`**: Initializes the key comparison function object.

#### Why This Is Used:
The constructor ensures that the map is in a valid state when it is created.

---

### **6. Destructor**
```cpp
~UnorderedMap() {
    clear();
}
```

#### What It Does:
This is the destructor for the `UnorderedMap` class. It cleans up resources when the map is destroyed.

#### Explanation:
- **`clear()`**: Calls the `clear` function to remove all elements from the map.

#### Why This Is Used:
The destructor ensures that the map’s resources are properly released when it is no longer needed.

---

### **7. Capacity Functions**
```cpp
bool empty() const {
    return num_elements == 0;
}

size_t size() const {
    return num_elements;
}

size_t max_size() const {
    return std::numeric_limits<size_t>::max();
}
```

#### What It Does:
These functions provide information about the map’s size and capacity.

#### Explanation:
- **`empty()`**: Returns `true` if the map is empty (i.e., `num_elements == 0`).
- **`size()`**: Returns the number of key-value pairs in the map.
- **`max_size()`**: Returns the maximum possible number of elements the map can hold.

#### Why These Are Used:
These functions allow users to query the map’s state, which is useful for debugging and logic control.

---

### **8. Iterator Class**
```cpp
class iterator {
private:
    UnorderedMap *map_ptr;
    size_t bucket_idx;
    typename std::list<std::pair<const Key, T>>::iterator bucket_iter;

    void advance_to_valid() {
        while (map_ptr && bucket_idx < map_ptr->buckets.size() &&
               bucket_iter == map_ptr->buckets[bucket_idx].end()) {
            ++bucket_idx;
            if (bucket_idx < map_ptr->buckets.size())
                bucket_iter = map_ptr->buckets[bucket_idx].begin();
        }
    }
```

#### What It Does:
This is the iterator class for the `UnorderedMap`. It allows users to traverse the map’s key-value pairs.

#### Explanation:
- **`map_ptr`**: A pointer to the `UnorderedMap` object.
- **`bucket_idx`**: The current bucket index.
- **`bucket_iter`**: An iterator for the current bucket’s list.
- **`advance_to_valid()`**: Skips empty buckets and ensures the iterator points to a valid element.

#### Why This Is Used:
Iterators are essential for traversing the map’s elements. The `advance_to_valid` function ensures that the iterator skips empty buckets, making traversal efficient.

---

### **9. Main Function**
```cpp
int main() {
    UnorderedMap<int, std::string> umap;
    umap.insert(std::pair<const int, std::string>(1, "one"));
    umap.insert(std::pair<const int, std::string>(2, "two"));
    umap[3] = "three";

    std::cout << "Size: " << umap.size() << std::endl;
    std::cout << "Key 2: " << umap.at(2) << std::endl;

    for (UnorderedMap<int, std::string>::iterator it = umap.begin(); it != umap.end(); ++it) {
        std::cout << it->first << " : " << it->second << std::endl;
    }

    umap.erase(1);
    std::cout << "After erasing key 1, size: " << umap.size() << std::endl;

    umap.clear();
    std::cout << "After clearing, empty: " << (umap.empty() ? "true" : "false") << std::endl;

    return 0;
}
```

#### What It Does:
This is the main function that demonstrates how to use the `UnorderedMap` class.

#### Explanation:
- **`UnorderedMap<int, std::string> umap;`**: Creates an `UnorderedMap` that maps integers to strings.
- **`umap.insert(...)`**: Inserts key-value pairs into the map.
- **`umap[3] = "three";`**: Uses the `operator[]` to insert a key-value pair.
- **`umap.size()`**: Prints the number of elements in the map.
- **`umap.at(2)`**: Accesses the value associated with key `2`.
- **Iterator Loop**: Traverses and prints all key-value pairs in the map.
- **`umap.erase(1)`**: Removes the key-value pair with key `1`.
- **`umap.clear()`**: Removes all elements from the map.

#### Why This Is Used:
The main function demonstrates the map’s functionality and ensures that it works as expected.

---

### **Summary**
This code implements a custom hash table (`UnorderedMap`) using a vector of lists for chaining. It provides efficient storage, retrieval, and traversal of key-value pairs. The use of templates makes it generic, and the iterator class allows for easy traversal. The main function demonstrates how to use the map in practice.