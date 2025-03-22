# Step-by-Step Explanation: main.cpp

Let’s dive into the code step by step, breaking it down into manageable sections and explaining everything in detail. I’ll start from the top and work our way down, ensuring that every concept is clear and well-explained.

---

### **1. Header Files**
```cpp
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <functional>
#include <stdexcept>
#include <utility>
#include <initializer_list>
#include <cmath> // Added for std::ceil
```

#### **What it does:**
These are **header files** that provide functionality used in the code. Think of them as toolboxes that contain tools (functions, classes, etc.) we need to build our program.

#### **Breakdown:**
- `<iostream>`: For input/output operations (e.g., `std::cout`).
- `<string>`: For working with strings (e.g., `std::string`).
- `<vector>`: For dynamic arrays (e.g., `std::vector`).
- `<list>`: For linked lists (e.g., `std::list`).
- `<functional>`: For function objects (e.g., `std::hash`).
- `<stdexcept>`: For exception handling (e.g., `std::runtime_error`).
- `<utility>`: For utility functions (e.g., `std::pair`).
- `<initializer_list>`: For initializing objects with lists (e.g., `{1, 2, 3}`).
- `<cmath>`: For mathematical functions (e.g., `std::ceil`).

#### **Why they are used:**
These headers provide the building blocks for the hash table. For example:
- `std::vector` is used to store the buckets.
- `std::list` is used to handle collisions in each bucket.
- `std::hash` is used to compute hash codes for keys.

---

### **2. Template Declaration**
```cpp
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class UnorderedMap {
```

#### **What it does:**
This declares a **template class** called `UnorderedMap`. A template is like a blueprint for a class that can work with any data type.

#### **Breakdown:**
- `typename Key`: The type of the keys (e.g., `std::string`).
- `typename Value`: The type of the values (e.g., `int`).
- `typename Hash = std::hash<Key>`: The hash function to use (default is `std::hash`).

#### **Why it’s used:**
Templates make the class **generic**, meaning it can work with any key and value types (e.g., `UnorderedMap<std::string, int>` or `UnorderedMap<int, double>`).

---

### **3. Private Members**
```cpp
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
```

#### **What it does:**
These are the **private members** of the class, which are only accessible within the class.

#### **Breakdown:**
1. **KeyValuePair**:
   - A structure to store a key and its corresponding value.
   - Example: If the key is `"apple"` and the value is `5`, this structure holds `{"apple", 5}`.

2. **Bucket**:
   - A type alias for a linked list (`std::list`) of `KeyValuePair` objects.
   - Each bucket is a linked list that stores key-value pairs that hash to the same index.

3. **buckets**:
   - A vector of buckets. This is the **hash table** itself.
   - Example: If there are 16 buckets, `buckets` is a vector of 16 linked lists.

4. **hasher**:
   - The hash function object (default is `std::hash`).
   - Example: `hasher("apple")` computes a hash code for the key `"apple"`.

5. **element_count**:
   - The number of key-value pairs in the map.

6. **max_load_factor_value**:
   - The maximum load factor (ratio of elements to buckets) before rehashing occurs.

#### **Why they are used:**
- `KeyValuePair` stores the actual data.
- `Bucket` handles collisions by storing multiple key-value pairs in the same bucket.
- `buckets` is the core data structure of the hash table.
- `hasher` computes the bucket index for a key.
- `element_count` and `max_load_factor_value` help manage the size and performance of the hash table.

---

### **4. Private Methods**
```cpp
private:
    size_t getBucketIndex(const Key& key) const {
        size_t hash_code = hasher(key);
        return hash_code % buckets.size();
    }
    
    typename Bucket::iterator findInBucket(Bucket& bucket, const Key& key) {
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (it->key == key) {
                return it;
            }
        }
        return bucket.end();
    }
    
    typename Bucket::const_iterator findInBucket(const Bucket& bucket, const Key& key) const {
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (it->key == key) {
                return it;
            }
        }
        return bucket.end();
    }
    
    void rehashImpl(size_t new_bucket_count) {
        std::vector<Bucket> new_buckets(new_bucket_count);
        
        for (const Bucket& bucket : buckets) {
            for (const KeyValuePair& pair : bucket) {
                size_t new_index = hasher(pair.key) % new_bucket_count;
                new_buckets[new_index].push_back(pair);
            }
        }
        
        buckets.swap(new_buckets);
    }
```

#### **What it does:**
These are helper methods used internally by the class.

#### **Breakdown:**
1. **getBucketIndex**:
   - Computes the bucket index for a given key.
   - Example: If `hasher("apple")` returns `123` and there are 16 buckets, `123 % 16 = 11`, so the bucket index is `11`.

2. **findInBucket**:
   - Searches for a key in a specific bucket.
   - Example: If the bucket contains `[{"apple", 5}, {"banana", 8}]`, searching for `"apple"` returns an iterator to `{"apple", 5}`.

3. **rehashImpl**:
   - Resizes the hash table and redistributes the elements.
   - Example: If the current bucket count is 16 and `new_bucket_count` is 32, all elements are moved to the new buckets.

#### **Why they are used:**
- `getBucketIndex` ensures keys are distributed evenly across buckets.
- `findInBucket` helps locate keys within a bucket.
- `rehashImpl` maintains performance by resizing the hash table when it becomes too full.

---

### **5. Public Methods**
```cpp
public:
    explicit UnorderedMap(size_t bucket_count = 16) 
        : buckets(bucket_count), element_count(0), max_load_factor_value(1.0) {}
    
    UnorderedMap(std::initializer_list<std::pair<Key, Value>> init, size_t bucket_count = 16)
        : buckets(bucket_count), element_count(0), max_load_factor_value(1.0) {
        for (const auto& pair : init) {
            insert(pair.first, pair.second);
        }
    }
    
    UnorderedMap(const UnorderedMap& other) = default;
    UnorderedMap(UnorderedMap&& other) noexcept = default;
    UnorderedMap& operator=(const UnorderedMap& other) = default;
```

#### **What it does:**
These are **constructors** and **assignment operators** for the class.

#### **Breakdown:**
1. **Default Constructor**:
   - Initializes the map with a default bucket count (16).
   - Example: `UnorderedMap<std::string, int> map;`.

2. **Initializer List Constructor**:
   - Initializes the map with a list of key-value pairs.
   - Example: `UnorderedMap<std::string, int> map = {{"apple", 5}, {"banana", 8}};`.

3. **Copy/Move Constructors and Assignment Operators**:
   - Handle copying and moving the map.

#### **Why they are used:**
- Constructors initialize the map.
- Initializer lists make it easy to create a map with initial data.
- Copy/move operations ensure the map can be safely copied or moved.

---

### **6. Main Function**
```cpp
int main() {
    UnorderedMap<std::string, int> map;
    
    map.insert("apple", 5);
    map.insert("banana", 8);
    map.insert("orange", 10);
    
    std::cout << "apple: " << map["apple"] << std::endl;
    
    map["grape"] = 12;
    
    for (auto it = map.begin(); it != map.end(); ++it) {
        std::cout << (*it).first << ": " << (*it).second << std::endl;
    }
    
    std::cout << "Contains 'pear': " << (map.contains("pear") ? "yes" : "no") << std::endl;
    
    std::cout << "Size: " << map.size() << std::endl;
    std::cout << "Bucket count: " << map.bucket_count() << std::endl;
    
    map.erase("banana");
}
```

#### **What it does:**
This demonstrates how to use the `UnorderedMap` class.

#### **Breakdown:**
1. **Create a Map**:
   - `UnorderedMap<std::string, int> map;` creates a map with string keys and integer values.

2. **Insert Elements**:
   - `map.insert("apple", 5);` adds `{"apple", 5}` to the map.

3. **Access Elements**:
   - `map["apple"]` retrieves the value for `"apple"`.

4. **Iterate Over Elements**:
   - The `for` loop prints all key-value pairs.

5. **Check for Key**:
   - `map.contains("pear")` checks if `"pear"` is in the map.

6. **Get Size and Bucket Count**:
   - `map.size()` returns the number of elements.
   - `map.bucket_count()` returns the number of buckets.

7. **Remove an Element**:
   - `map.erase("banana");` removes `"banana"` from the map.

#### **Why it’s used:**
This demonstrates the functionality of the `UnorderedMap` class in a real-world scenario.

---

### **Summary**
This code implements a hash table from scratch, using templates to make it generic and efficient. It handles collisions with chaining, dynamically resizes the table, and provides iterators for traversal. The main function demonstrates how to use the class, making it a complete and educational example.