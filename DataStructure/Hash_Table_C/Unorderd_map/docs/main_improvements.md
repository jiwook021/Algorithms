# Suggested Improvements: main.cpp

Here are several **improvements** that could be made to the code, categorized by **performance**, **readability**, **maintainability**, **error handling**, and **best practices**. Each suggestion includes an explanation of **why** it is an improvement and **how** it could be implemented.

---

### **1. Dynamic Resizing (Performance Improvement)**

#### **Why:**
The current implementation uses a fixed number of buckets (`default_bucket_count = 8`). If the number of elements grows significantly, the hash table will become inefficient due to increased collisions. Dynamic resizing ensures that the number of buckets grows as the number of elements increases, maintaining **O(1)** average-time complexity for operations.

#### **How:**
Add a **load factor** threshold (e.g., 0.75). When the load factor (number of elements / number of buckets) exceeds this threshold, resize the hash table by doubling the number of buckets and rehashing all elements.

```cpp
void rehash(size_t new_bucket_count) {
    std::vector<std::list<std::pair<const Key, T>>> new_buckets(new_bucket_count);
    for (auto &bucket : buckets) {
        for (auto &pair : bucket) {
            size_t new_index = hash_func(pair.first) % new_bucket_count;
            new_buckets[new_index].push_back(pair);
        }
    }
    buckets = std::move(new_buckets);
}

void check_and_rehash() {
    double load_factor = static_cast<double>(num_elements) / buckets.size();
    if (load_factor > 0.75) {
        rehash(buckets.size() * 2);
    }
}
```

Call `check_and_rehash()` after every insertion.

---

### **2. Const-Correctness (Best Practice)**

#### **Why:**
The current implementation lacks `const` qualifiers for member functions that do not modify the object (e.g., `bucket_index`, `empty`, `size`). Adding `const` ensures that these functions can be called on `const` objects, improving safety and readability.

#### **How:**
Add `const` qualifiers to appropriate member functions:

```cpp
size_t bucket_index(const Key &key) const {
    return hash_func(key) % buckets.size();
}

bool empty() const {
    return num_elements == 0;
}

size_t size() const {
    return num_elements;
}
```

---

### **3. Error Handling (Robustness)**

#### **Why:**
The current implementation does not handle errors gracefully. For example, accessing a non-existent key using `at()` could throw an exception, but the exception message is not descriptive.

#### **How:**
Improve error handling by providing meaningful exception messages:

```cpp
T &at(const Key &key) {
    size_t index = bucket_index(key);
    for (auto &pair : buckets[index]) {
        if (key_equal_obj(pair.first, key)) {
            return pair.second;
        }
    }
    throw std::out_of_range("Key not found in UnorderedMap");
}
```

---

### **4. Move Semantics (Performance Improvement)**

#### **Why:**
The current implementation does not take advantage of move semantics, which can improve performance by avoiding unnecessary copies of objects.

#### **How:**
Add move constructors and move assignment operators:

```cpp
UnorderedMap(UnorderedMap &&other) noexcept
    : buckets(std::move(other.buckets)),
      num_elements(other.num_elements),
      hash_func(std::move(other.hash_func)),
      key_equal_obj(std::move(other.key_equal_obj)) {
    other.num_elements = 0;
}

UnorderedMap &operator=(UnorderedMap &&other) noexcept {
    if (this != &other) {
        buckets = std::move(other.buckets);
        num_elements = other.num_elements;
        hash_func = std::move(other.hash_func);
        key_equal_obj = std::move(other.key_equal_obj);
        other.num_elements = 0;
    }
    return *this;
}
```

---

### **5. Iterator Improvements (Readability and Maintainability)**

#### **Why:**
The current iterator implementation is incomplete and lacks some standard iterator functionality (e.g., `operator->`, `operator==`, `operator!=`).

#### **How:**
Add missing iterator functionality:

```cpp
class iterator {
public:
    using value_type = std::pair<const Key, T>;
    using reference = value_type &;
    using pointer = value_type *;

    iterator(UnorderedMap *map_ptr = nullptr, size_t bucket_idx = 0,
             typename std::list<value_type>::iterator bucket_iter = {})
        : map_ptr(map_ptr), bucket_idx(bucket_idx), bucket_iter(bucket_iter) {
        if (map_ptr) advance_to_valid();
    }

    reference operator*() const { return *bucket_iter; }
    pointer operator->() const { return &(*bucket_iter); }

    iterator &operator++() {
        ++bucket_iter;
        advance_to_valid();
        return *this;
    }

    bool operator==(const iterator &other) const {
        return map_ptr == other.map_ptr &&
               bucket_idx == other.bucket_idx &&
               bucket_iter == other.bucket_iter;
    }

    bool operator!=(const iterator &other) const {
        return !(*this == other);
    }

private:
    void advance_to_valid() {
        while (map_ptr && bucket_idx < map_ptr->buckets.size() &&
               bucket_iter == map_ptr->buckets[bucket_idx].end()) {
            ++bucket_idx;
            if (bucket_idx < map_ptr->buckets.size())
                bucket_iter = map_ptr->buckets[bucket_idx].begin();
        }
    }

    UnorderedMap *map_ptr;
    size_t bucket_idx;
    typename std::list<value_type>::iterator bucket_iter;
};
```

---

### **6. Documentation (Maintainability)**

#### **Why:**
The current code lacks detailed comments and documentation, making it harder for others (or yourself in the future) to understand and maintain.

#### **How:**
Add detailed comments and documentation:

```cpp
/**
 * UnorderedMap is a custom hash table implementation that stores key-value pairs.
 * It uses chaining (std::list) for collision resolution and provides O(1) average-time
 * complexity for insertion, deletion, and lookup operations.
 *
 * @tparam Key The type of the keys.
 * @tparam T The type of the values.
 * @tparam Hash The hash function type (default: std::hash<Key>).
 * @tparam KeyEqual The key comparison function type (default: std::equal_to<Key>).
 */
template <typename Key, typename T, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class UnorderedMap {
    // Class implementation...
};
```

---

### **7. Unit Tests (Robustness)**

#### **Why:**
The current implementation lacks unit tests, making it difficult to verify correctness and catch edge cases.

#### **How:**
Add unit tests using a framework like **Google Test**:

```cpp
#include <gtest/gtest.h>

TEST(UnorderedMapTest, InsertAndAccess) {
    UnorderedMap<int, std::string> umap;
    umap.insert({1, "one"});
    EXPECT_EQ(umap.at(1), "one");
}

TEST(UnorderedMapTest, Erase) {
    UnorderedMap<int, std::string> umap;
    umap.insert({1, "one"});
    umap.erase(1);
    EXPECT_TRUE(umap.empty());
}

// Add more tests for edge cases, resizing, etc.
```

---

### **8. Use of `std::unordered_map` Features (Best Practice)**

#### **Why:**
The current implementation is a simplified version of `std::unordered_map`. To align with the standard library, consider adding missing features like `bucket_count`, `load_factor`, and `reserve`.

#### **How:**
Add these features:

```cpp
size_t bucket_count() const {
    return buckets.size();
}

double load_factor() const {
    return static_cast<double>(num_elements) / buckets.size();
}

void reserve(size_t count) {
    rehash(count);
}
```

---

### **Summary of Improvements**
| **Category**       | **Improvement**               | **Why**                                                                 | **How**                                                                 |
|---------------------|-------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Performance         | Dynamic resizing              | Maintains O(1) average-time complexity                                  | Add `rehash` and `check_and_rehash` functions                           |
| Best Practice       | Const-correctness             | Improves safety and readability                                         | Add `const` qualifiers to member functions                              |
| Robustness          | Error handling                | Provides meaningful error messages                                      | Add descriptive exception messages                                      |
| Performance         | Move semantics                | Avoids unnecessary copies                                               | Add move constructor and move assignment operator                       |
| Readability         | Iterator improvements         | Makes iterator functionality complete and standard-compliant            | Add missing iterator operators                                          |
| Maintainability     | Documentation                 | Makes code easier to understand and maintain                            | Add detailed comments and documentation                                 |
| Robustness          | Unit tests                    | Verifies correctness and catches edge cases                             | Add unit tests using a framework like Google Test                       |
| Best Practice       | Standard library alignment    | Aligns with `std::unordered_map` features                               | Add `bucket_count`, `load_factor`, and `reserve` functions              |

By implementing these improvements, the code will be more **efficient**, **readable**, **maintainable**, and **robust**.