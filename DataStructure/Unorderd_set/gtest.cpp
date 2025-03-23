#include <iostream>
#include <vector>
#include <list>
#include <functional>
#include <algorithm>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <gtest/gtest.h>

/**
 * A thread-safe list implementation
 * 
 * Time complexity:
 *   - Insert at front: O(1)
 *   - Insert at back: O(1) with tail pointer
 *   - Remove: O(n) as we need to find the element
 *   - Find: O(n)
 * 
 * Space complexity: O(n) where n is the number of elements
 */
template<typename T>
class List {
private:
    // Node structure representing each element in the list
    struct Node {
        T data;
        std::shared_ptr<Node> next;
        
        // Constructor
        Node(const T& value) : data(value), next(nullptr) {}
    };
    
    std::shared_ptr<Node> head;     // Points to first node
    std::shared_ptr<Node> tail;     // Points to last node for O(1) insertion at back
    size_t count;                   // Number of elements in the list
    
    // Mutex for thread safety
    mutable std::mutex mutex;       // mutable allows locking in const methods
    
public:
    // Iterator implementation for traversing the list
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;
        
        // Constructor
        Iterator(std::shared_ptr<Node> ptr) : current(ptr) {}
        
        // Dereference operator
        reference operator*() const {
            return current->data;
        }
        
        // Arrow operator
        pointer operator->() const {
            return &(current->data);
        }
        
        // Pre-increment
        Iterator& operator++() {
            current = current->next;
            return *this;
        }
        
        // Post-increment
        Iterator operator++(int) {
            Iterator tmp = *this;
            current = current->next;
            return tmp;
        }
        
        // Equality operators
        bool operator==(const Iterator& other) const {
            return current == other.current;
        }
        
        bool operator!=(const Iterator& other) const {
            return current != other.current;
        }
        
    private:
        std::shared_ptr<Node> current; // Current node pointer
    };
    
    // Const iterator implementation for read-only traversal
    class ConstIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = const T*;
        using reference = const T&;
        
        // Constructor
        ConstIterator(std::shared_ptr<Node> ptr) : current(ptr) {}
        
        // Dereference operator
        reference operator*() const {
            return current->data;
        }
        
        // Arrow operator
        pointer operator->() const {
            return &(current->data);
        }
        
        // Pre-increment
        ConstIterator& operator++() {
            current = current->next;
            return *this;
        }
        
        // Post-increment
        ConstIterator operator++(int) {
            ConstIterator tmp = *this;
            current = current->next;
            return tmp;
        }
        
        // Equality operators
        bool operator==(const ConstIterator& other) const {
            return current == other.current;
        }
        
        bool operator!=(const ConstIterator& other) const {
            return current != other.current;
        }
        
    private:
        std::shared_ptr<Node> current; // Current node pointer
    };
    
    // Default constructor
    List() : head(nullptr), tail(nullptr), count(0) {}
    
    // Initialize with initializer list
    List(std::initializer_list<T> il) : head(nullptr), tail(nullptr), count(0) {
        for (const auto& value : il) {
            push_back(value);
        }
    }
    
    // Copy constructor - creates a deep copy
    List(const List& other) : head(nullptr), tail(nullptr), count(0) {
        std::lock_guard<std::mutex> lock(other.mutex); // Lock required to safely copy the other list
        auto current = other.head;
        while (current) {
            push_back(current->data);
            current = current->next;
        }
    }
    
    // Move constructor
    List(List&& other) noexcept : head(nullptr), tail(nullptr), count(0) {
        std::lock_guard<std::mutex> lock(other.mutex); // Lock required for thread safety
        head = std::move(other.head);
        tail = std::move(other.tail);
        count = other.count;
        other.count = 0;
    }
    
    // Destructor - shared_ptr will automatically clean up nodes
    ~List() = default;
    
    // Copy assignment operator
    List& operator=(const List& other) {
        if (this != &other) {
            List temp(other);  // Create a copy
            std::lock_guard<std::mutex> lock(mutex); // Lock required for thread safety when updating this list
            head = std::move(temp.head);
            tail = std::move(temp.tail);
            count = temp.count;
        }
        return *this;
    }
    
    // Move assignment operator
    List& operator=(List&& other) noexcept {
        if (this != &other) {
            std::lock(mutex, other.mutex); // Lock both mutexes to prevent deadlocks
            std::lock_guard<std::mutex> lock_this(mutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock_other(other.mutex, std::adopt_lock);
            
            head = std::move(other.head);
            tail = std::move(other.tail);
            count = other.count;
            other.count = 0;
        }
        return *this;
    }
    
    // Add element to the front - O(1)
    void push_front(const T& value) {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe modification
        auto new_node = std::make_shared<Node>(value);
        if (!head) {
            head = new_node;
            tail = new_node;
        } else {
            new_node->next = head;
            head = new_node;
        }
        count++;
    }
    
    // Add element to the back - O(1) with tail pointer
    void push_back(const T& value) {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe modification
        auto new_node = std::make_shared<Node>(value);
        if (!head) {
            head = new_node;
            tail = new_node;
        } else {
            tail->next = new_node;
            tail = new_node;
        }
        count++;
    }
    
    // Remove first element - O(1)
    void pop_front() {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe modification
        if (!head) return;
        
        head = head->next;
        if (!head) tail = nullptr;
        count--;
    }
    
    // Remove element with given value - O(n)
    bool remove(const T& value) {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe modification
        if (!head) return false;
        
        // Special case: remove from head
        if (head->data == value) {
            head = head->next;
            if (!head) tail = nullptr;
            count--;
            return true;
        }
        
        // Search for the value
        auto current = head;
        while (current->next && !(current->next->data == value)) {
            current = current->next;
        }
        
        // If found, remove it
        if (current->next) {
            if (current->next == tail) {
                tail = current;
            }
            current->next = current->next->next;
            count--;
            return true;
        }
        
        return false;
    }
    
    // Check if list contains an element - O(n)
    bool contains(const T& value) const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        auto current = head;
        while (current) {
            if (current->data == value) return true;
            current = current->next;
        }
        return false;
    }
    
    // Get size of list - O(1)
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        return count;
    }
    
    // Check if list is empty - O(1)
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        return count == 0;
    }
    
    // Clear the list - O(1) with shared_ptr
    void clear() {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe modification
        head = nullptr;
        tail = nullptr;
        count = 0;
    }
    
    // Iterator methods
    Iterator begin() {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe access
        return Iterator(head);
    }
    
    Iterator end() {
        return Iterator(nullptr);
    }
    
    ConstIterator begin() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe access
        return ConstIterator(head);
    }
    
    ConstIterator end() const {
        return ConstIterator(nullptr);
    }
    
    ConstIterator cbegin() const {
        return begin();
    }
    
    ConstIterator cend() const {
        return end();
    }
};

/**
 * A thread-safe unordered_set implementation
 * 
 * Time complexity:
 *   - Insert: Average O(1), Worst O(n)
 *   - Find: Average O(1), Worst O(n)
 *   - Erase: Average O(1), Worst O(n)
 * 
 * Space complexity: O(n) where n is the number of elements
 */
template<typename T, typename Hash = std::hash<T>, typename Equal = std::equal_to<T>>
class SimpleUnorderedSet {
private:
    // Use std::list for buckets instead of manual linked list
    std::vector<std::list<T>> buckets;
    size_t element_count = 0;
    Hash hasher;
    Equal equal;
    float max_load_factor_val = 1.0f;
    
    // Mutex for thread safety
    mutable std::mutex mutex;

    // Iterator implementation
    template<bool IsConst>
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = typename std::conditional<IsConst, const T*, T*>::type;
        using reference = typename std::conditional<IsConst, const T&, T&>::type;
        using bucket_iterator = typename std::conditional<IsConst, 
                                typename std::list<T>::const_iterator, 
                                typename std::list<T>::iterator>::type;
        using bucket_vector = typename std::conditional<IsConst, 
                               const std::vector<std::list<T>>, 
                               std::vector<std::list<T>>>::type;

        Iterator() = default;

        Iterator(bucket_vector* buckets, size_t bucket_idx, bucket_iterator list_it)
            : buckets_(buckets), bucket_idx_(bucket_idx), list_it_(list_it) {
            // If we're pointing to the end of a bucket, find the next non-empty bucket
            skip_empty_buckets();
        }

        reference operator*() const {
            return *list_it_;
        }

        pointer operator->() const {
            return &(*list_it_);
        }

        // Pre-increment
        Iterator& operator++() {
            ++list_it_;
            // If we've reached the end of the current bucket, move to the next bucket
            if (list_it_ == (*buckets_)[bucket_idx_].end()) {
                ++bucket_idx_;
                skip_empty_buckets();
            }
            return *this;
        }

        // Post-increment
        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            if (bucket_idx_ >= buckets_->size() && other.bucket_idx_ >= buckets_->size())
                return true; // Both are end iterators
            return bucket_idx_ == other.bucket_idx_ && list_it_ == other.list_it_;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        bucket_vector* buckets_ = nullptr;
        size_t bucket_idx_ = 0;
        bucket_iterator list_it_ = bucket_iterator();

        // Skip empty buckets
        void skip_empty_buckets() {
            while (bucket_idx_ < buckets_->size() && (*buckets_)[bucket_idx_].empty()) {
                ++bucket_idx_;
            }
            if (bucket_idx_ < buckets_->size()) {
                list_it_ = (*buckets_)[bucket_idx_].begin();
            }
        }
    };

public:
    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;

    // Constructor
    explicit SimpleUnorderedSet(size_t bucket_count = 16)
        : buckets(bucket_count) {}

    // Initializer list constructor
    SimpleUnorderedSet(std::initializer_list<T> il, size_t bucket_count = 16)
        : buckets(bucket_count) {
        std::cout << "Initializer list constructor called with " << il.size() << " elements" << std::endl;
        for (const auto& value : il) {
            insert(value);
        }
    }

    // Iterator methods
    iterator begin() {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe access
        if (empty()) return end();
        return iterator(&buckets, 0, buckets[0].begin());
    }

    const_iterator begin() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe access
        if (empty()) return end();
        return const_iterator(&buckets, 0, buckets[0].begin());
    }

    const_iterator cbegin() const {
        return begin();
    }

    iterator end() {
        // No lock needed here as it returns a constant end iterator
        return iterator(&buckets, buckets.size(), typename std::list<T>::iterator());
    }

    const_iterator end() const {
        // No lock needed here as it returns a constant end iterator
        return const_iterator(&buckets, buckets.size(), typename std::list<T>::const_iterator());
    }

    const_iterator cend() const {
        return end();
    }

    // Capacity
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        return element_count == 0;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        return element_count;
    }

    // Modifiers
    std::pair<iterator, bool> insert(const T& value) {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe modification
        std::cout << "Inserting value " << value << std::endl;
        
        if (load_factor() > max_load_factor_val) {
            rehash(buckets.size() * 2);
        }

        size_t idx = bucket(value);
        auto& list = buckets[idx];

        // Check if value already exists
        auto it = std::find_if(list.begin(), list.end(), 
                               [&](const T& elem) { return equal(elem, value); });
        if (it != list.end()) {
            std::cout << "Value " << value << " already exists" << std::endl;
            return {iterator(&buckets, idx, it), false};
        }

        // Insert value
        list.push_back(value);
        ++element_count;
        auto list_it = --list.end(); // Iterator to the newly inserted element
        std::cout << "Value " << value << " inserted successfully" << std::endl;
        return {iterator(&buckets, idx, list_it), true};
    }

    size_t erase(const T& value) {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe modification
        size_t idx = bucket(value);
        auto& list = buckets[idx];

        auto it = std::find_if(list.begin(), list.end(), 
                               [&](const T& elem) { return equal(elem, value); });
        if (it != list.end()) {
            list.erase(it);
            --element_count;
            return 1;
        }
        return 0;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe modification
        for (auto& list : buckets) {
            list.clear();
        }
        element_count = 0;
    }

    // Lookup
    bool contains(const T& value) const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        std::cout << "Checking if contains value " << value << std::endl;
        size_t idx = bucket(value);
        const auto& list = buckets[idx];

        return std::any_of(list.begin(), list.end(), 
                          [&](const T& elem) { return equal(elem, value); });
    }

    iterator find(const T& value) {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        size_t idx = bucket(value);
        auto& list = buckets[idx];

        auto it = std::find_if(list.begin(), list.end(), 
                               [&](const T& elem) { return equal(elem, value); });
        if (it != list.end()) {
            return iterator(&buckets, idx, it);
        }
        return end();
    }

    const_iterator find(const T& value) const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        size_t idx = bucket(value);
        const auto& list = buckets[idx];

        auto it = std::find_if(list.begin(), list.end(), 
                               [&](const T& elem) { return equal(elem, value); });
        if (it != list.end()) {
            return const_iterator(&buckets, idx, it);
        }
        return end();
    }

    // Bucket interface
    size_t bucket_count() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        return buckets.size();
    }

    size_t bucket(const T& value) const {
        // No mutex needed here as it's a pure calculation using value
        return hasher(value) % buckets.size();
    }

    // Hash policy
    float load_factor() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        return static_cast<float>(element_count) / buckets.size();
    }

    float max_load_factor() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe read
        return max_load_factor_val;
    }

    void max_load_factor(float ml) {
        std::lock_guard<std::mutex> lock(mutex); // Lock required for thread-safe modification
        max_load_factor_val = ml;
    }

    void rehash(size_t count) {
        // Already inside a lock in insert() so no additional lock needed
        std::cout << "Rehashing to " << count << " buckets" << std::endl;
        if (count <= buckets.size()) return;

        std::vector<std::list<T>> new_buckets(count);
        
        // Move elements to new buckets
        for (auto& list : buckets) {
            for (auto& value : list) {
                size_t new_idx = hasher(value) % count;
                new_buckets[new_idx].push_back(value);
            }
        }
        
        buckets = std::move(new_buckets);
    }
};

// Google Test cases for List class
class ListTest : public ::testing::Test {
protected:
    List<int> list;
    
    void SetUp() override {
        // Initialize list with some values
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
    }
};

TEST_F(ListTest, InitialSize) {
    EXPECT_EQ(list.size(), 3);
    EXPECT_FALSE(list.empty());
}

TEST_F(ListTest, PushFront) {
    list.push_front(5);
    EXPECT_EQ(list.size(), 4);
    EXPECT_EQ(*list.begin(), 5);
}

TEST_F(ListTest, PushBack) {
    list.push_back(40);
    EXPECT_EQ(list.size(), 4);
    
    // Find the last element
    auto it = list.begin();
    for (size_t i = 0; i < 3; ++i) ++it;
    EXPECT_EQ(*it, 40);
}

TEST_F(ListTest, PopFront) {
    list.pop_front();
    EXPECT_EQ(list.size(), 2);
    EXPECT_EQ(*list.begin(), 20);
}

TEST_F(ListTest, Remove) {
    EXPECT_TRUE(list.remove(20));
    EXPECT_EQ(list.size(), 2);
    EXPECT_FALSE(list.contains(20));
    
    // Remove non-existent element
    EXPECT_FALSE(list.remove(50));
}

TEST_F(ListTest, Contains) {
    EXPECT_TRUE(list.contains(10));
    EXPECT_TRUE(list.contains(20));
    EXPECT_TRUE(list.contains(30));
    EXPECT_FALSE(list.contains(40));
}

TEST_F(ListTest, Clear) {
    list.clear();
    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.size(), 0);
}

TEST_F(ListTest, InitializerList) {
    List<int> new_list = {1, 2, 3, 4, 5};
    EXPECT_EQ(new_list.size(), 5);
    EXPECT_TRUE(new_list.contains(1));
    EXPECT_TRUE(new_list.contains(5));
}

TEST_F(ListTest, CopyConstructor) {
    List<int> copy(list);
    EXPECT_EQ(copy.size(), 3);
    EXPECT_TRUE(copy.contains(10));
    EXPECT_TRUE(copy.contains(20));
    EXPECT_TRUE(copy.contains(30));
}

TEST_F(ListTest, MoveConstructor) {
    List<int> moved(std::move(list));
    EXPECT_EQ(moved.size(), 3);
    EXPECT_TRUE(moved.contains(10));
    EXPECT_TRUE(moved.contains(20));
    EXPECT_TRUE(moved.contains(30));
    EXPECT_TRUE(list.empty());
}

// Google Test cases for SimpleUnorderedSet class
class SimpleUnorderedSetTest : public ::testing::Test {
protected:
    SimpleUnorderedSet<int> set;
    
    void SetUp() override {
        // Initialize set with some values
        set.insert(10);
        set.insert(20);
        set.insert(30);
    }
};

TEST_F(SimpleUnorderedSetTest, InitialSize) {
    EXPECT_EQ(set.size(), 3);
    EXPECT_FALSE(set.empty());
}

TEST_F(SimpleUnorderedSetTest, Insert) {
    auto [it, inserted] = set.insert(40);
    EXPECT_TRUE(inserted);
    EXPECT_EQ(set.size(), 4);
    EXPECT_TRUE(set.contains(40));
    
    // Try inserting a duplicate
    auto [it2, inserted2] = set.insert(10);
    EXPECT_FALSE(inserted2);
    EXPECT_EQ(set.size(), 4);
}

TEST_F(SimpleUnorderedSetTest, Erase) {
    EXPECT_EQ(set.erase(20), 1);
    EXPECT_EQ(set.size(), 2);
    EXPECT_FALSE(set.contains(20));
    
    // Try erasing a non-existent element
    EXPECT_EQ(set.erase(50), 0);
}

TEST_F(SimpleUnorderedSetTest, Contains) {
    EXPECT_TRUE(set.contains(10));
    EXPECT_TRUE(set.contains(20));
    EXPECT_TRUE(set.contains(30));
    EXPECT_FALSE(set.contains(40));
}

TEST_F(SimpleUnorderedSetTest, Find) {
    auto it = set.find(20);
    EXPECT_NE(it, set.end());
    EXPECT_EQ(*it, 20);
    
    // Try finding a non-existent element
    auto it2 = set.find(50);
    EXPECT_EQ(it2, set.end());
}

TEST_F(SimpleUnorderedSetTest, Clear) {
    set.clear();
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0);
}

TEST_F(SimpleUnorderedSetTest, InitializerList) {
    SimpleUnorderedSet<int> new_set = {1, 2, 3, 4, 5};
    EXPECT_EQ(new_set.size(), 5);
    EXPECT_TRUE(new_set.contains(1));
    EXPECT_TRUE(new_set.contains(5));
}

TEST_F(SimpleUnorderedSetTest, LoadFactor) {
    EXPECT_LE(set.load_factor(), set.max_load_factor());
    
    // Add many elements to trigger rehash
    for (int i = 100; i < 150; ++i) {
        set.insert(i);
    }
    
    EXPECT_LE(set.load_factor(), set.max_load_factor());
}

// Edge cases test
TEST_F(SimpleUnorderedSetTest, EdgeCases) {
    // Empty set behavior
    SimpleUnorderedSet<int> empty_set;
    EXPECT_TRUE(empty_set.empty());
    EXPECT_EQ(empty_set.size(), 0);
    EXPECT_EQ(empty_set.find(10), empty_set.end());
    
    // Clear and reuse
    set.clear();
    EXPECT_TRUE(set.empty());
    set.insert(100);
    EXPECT_EQ(set.size(), 1);
    EXPECT_TRUE(set.contains(100));
}

// Test program using Google Test
int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Run all tests
    return RUN_ALL_TESTS();
}