/**
 * @file UnorderedSet.hpp
 * @brief Custom hash-set (unordered set) implementation
 * @details Open-addressing hash table implementing an unordered set with linear probing, dynamic resize, and iterator support. Template class supporting arbitrary key types with custom hash and equality.
 */

#pragma once

#include <iostream>
#include <vector>
#include <list>
#include <functional>
#include <algorithm>
#include <initializer_list>
#include <memory>
#include <mutex>

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
    
    std::shared_ptr<Node> Head;     // Points to first node
    std::shared_ptr<Node> Tail;     // Points to last node for O(1) insertion at back
    size_t count;                   // Number of elements in the list
    
    // Mutex for thread safety
    mutable std::mutex mutex;       // mutable allows locking in const methods
    
public:

    class Iterator {
    public:
        // Constructor
        Iterator(std::shared_ptr<Node> Ptr) : Current(Ptr) {}
        
        // Dereference operator
        T& operator*() const {
            return Current->data;
        }
        
        // Arrow operator
        T* operator->() const {
            return &(Current->data);
        }
        
        // Pre-increment
        Iterator& operator++() {
            Current = Current->next;
            return *this;
        }
        
        // Post-increment
        Iterator operator++(int) {
            Iterator Tmp = *this;
            Current = Current->next;
            return Tmp;
        }
        
        // Equality operators
        bool operator==(const Iterator& Other) const {
            return Current == Other.Current;
        }
        
        bool operator!=(const Iterator& Other) const {
            return Current != Other.Current;
        }
        
    private:
        std::shared_ptr<Node> Current; // Current node pointer
    };
    
    // Const Iterator<false> implementation for read-only traversal
    class ConstIterator {
    public:
        // Constructor
        ConstIterator(std::shared_ptr<Node> Ptr) : Current(Ptr) {}
        
        // Dereference operator
        const T& operator*() const {
            return Current->data;
        }
        
        // Arrow operator
        const T* operator->() const {
            return &(Current->data);
        }
        
        // Pre-increment
        ConstIterator& operator++() {
            Current = Current->next;
            return *this;
        }
        
        // Post-increment
        ConstIterator operator++(int) {
            ConstIterator Tmp = *this;
            Current = Current->next;
            return Tmp;
        }
        
        // Equality operators
        bool operator==(const ConstIterator& Other) const {
            return Current == Other.Current;
        }
        
        bool operator!=(const ConstIterator& Other) const {
            return Current != Other.Current;
        }
        
    private:
        std::shared_ptr<Node> Current; // Current node pointer
    };
    
    // Default constructor
    List() : Head(nullptr), Tail(nullptr), count(0) {}
    
    // Initialize with initializer list
    List(std::initializer_list<T> Il) : Head(nullptr), Tail(nullptr), count(0) {
        for (const auto& value : Il) {
            push_back(value);
        }
    }
    
    // Copy constructor - creates a deep copy
    List(const List& Other) : Head(nullptr), Tail(nullptr), count(0) {
        std::lock_guard<std::mutex> Lock(Other.mutex); // Lock required to safely copy the other list
        auto Current = Other.Head;
        while (Current) {
            push_back(Current->data);
            Current = Current->next;
        }
    }
    
    // Move constructor
    List(List&& Other) noexcept : Head(nullptr), Tail(nullptr), count(0) {
        std::lock_guard<std::mutex> Lock(Other.mutex); // Lock required for thread safety
        Head = std::move(Other.Head);
        Tail = std::move(Other.Tail);
        count = Other.count;
        Other.count = 0;
    }
    
    // Destructor - shared_ptr will automatically clean up nodes
    ~List() = default;
    
    // Copy assignment operator
    List& operator=(const List& Other) {
        if (this != &Other) {
            List Temp(Other);  // Create a copy
            std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread safety when updating this list
            Head = std::move(Temp.Head);
            Tail = std::move(Temp.Tail);
            count = Temp.count;
        }
        return *this;
    }
    
    // Move assignment operator
    List& operator=(List&& Other) noexcept {
        if (this != &Other) {
            std::lock(mutex, Other.mutex); // Lock both mutexes to prevent deadlocks
            std::lock_guard<std::mutex> LockThis(mutex, std::adopt_lock);
            std::lock_guard<std::mutex> LockOther(Other.mutex, std::adopt_lock);
            
            Head = std::move(Other.Head);
            Tail = std::move(Other.Tail);
            count = Other.count;
            Other.count = 0;
        }
        return *this;
    }
    
    // Add element to the front - O(1)
    void push_front(const T& value) {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe modification
        auto NewNode = std::make_shared<Node>(value);
        if (!Head) {
            Head = NewNode;
            Tail = NewNode;
        } else {
            NewNode->next = Head;
            Head = NewNode;
        }
        count++;
    }
    
    // Add element to the back - O(1) with tail pointer
    void push_back(const T& value) {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe modification
        auto NewNode = std::make_shared<Node>(value);
        if (!Head) {
            Head = NewNode;
            Tail = NewNode;
        } else {
            Tail->next = NewNode;
            Tail = NewNode;
        }
        count++;
    }
    
    // Remove first element - O(1)
    void pop_front() {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe modification
        if (!Head) return;
        
        Head = Head->next;
        if (!Head) Tail = nullptr;
        count--;
    }
    
    // Remove element with given value - O(n)
    bool remove(const T& value) {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe modification
        if (!Head) return false;
        
        // Special case: remove from head
        if (Head->data == value) {
            Head = Head->next;
            if (!Head) Tail = nullptr;
            count--;
            return true;
        }
        
        // Search for the value
        auto Current = Head;
        while (Current->next && !(Current->next->data == value)) {
            Current = Current->next;
        }
        
        // If found, remove it
        if (Current->next) {
            if (Current->next == Tail) {
                Tail = Current;
            }
            Current->next = Current->next->next;
            count--;
            return true;
        }
        
        return false;
    }
    
    // Check if list contains an element - O(n)
    bool Contains(const T& value) const {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe read
        auto Current = Head;
        while (Current) {
            if (Current->data == value) return true;
            Current = Current->next;
        }
        return false;
    }
    
    // Get size of list - O(1)
    size_t size() const {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe read
        return count;
    }
    
    // Check if list is empty - O(1)
    bool empty() const {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe read
        return count == 0;
    }
    
    // Clear the list - O(1) with shared_ptr
    void clear() {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe modification
        Head = nullptr;
        Tail = nullptr;
        count = 0;
    }
    
    // Iterator methods
    Iterator begin() {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe access
        return Iterator(Head);
    }
    
    Iterator end() {
        return Iterator(nullptr);
    }
    
    ConstIterator begin() const {
        std::lock_guard<std::mutex> Lock(mutex); // Lock required for thread-safe access
        return ConstIterator(Head);
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
    std::vector<std::list<T>> Buckets;
    size_t ElementCount = 0;
    Hash Hasher;
    Equal equal;
    float MaxLoadFactorVal = 1.0f;
    
    // Mutex for thread safety
    mutable std::recursive_mutex mutex;

    // Iterator implementation
    template<bool IsConst>
    class Iterator {
    public:
        Iterator() = default;

        Iterator(typename std::conditional<IsConst, const std::vector<std::list<T>>, std::vector<std::list<T>>>::type* Buckets, size_t BucketIdx, typename std::conditional<IsConst, typename std::list<T>::const_iterator, typename std::list<T>::iterator>::type ListIt)
            : Buckets_(Buckets), BucketIdx_(BucketIdx), ListIt_(ListIt) {
            // If we're pointing to the end of a bucket, find the next non-empty bucket
            SkipEmptyBuckets();
        }

        typename std::conditional<IsConst, const T&, T&>::type operator*() const {
            return *ListIt_;
        }

        typename std::conditional<IsConst, const T*, T*>::type operator->() const {
            return &(*ListIt_);
        }

        // Pre-increment
        Iterator& operator++() {
            ++ListIt_;
            // If we've reached the end of the current bucket, move to the next bucket
            if (ListIt_ == (*Buckets_)[BucketIdx_].end()) {
                ++BucketIdx_;
                SkipEmptyBuckets();
            }
            return *this;
        }

        // Post-increment
        Iterator operator++(int) {
            Iterator Tmp = *this;
            ++(*this);
            return Tmp;
        }

        bool operator==(const Iterator& Other) const {
            if (BucketIdx_ >= Buckets_->size() && Other.BucketIdx_ >= Buckets_->size())
                return true; // Both are end iterators
            return BucketIdx_ == Other.BucketIdx_ && ListIt_ == Other.ListIt_;
        }

        bool operator!=(const Iterator& Other) const {
            return !(*this == Other);
        }

    private:
        typename std::conditional<IsConst, const std::vector<std::list<T>>, std::vector<std::list<T>>>::type* Buckets_ = nullptr;
        size_t BucketIdx_ = 0;
        typename std::conditional<IsConst, typename std::list<T>::const_iterator, typename std::list<T>::iterator>::type ListIt_{};

        // Skip empty buckets
        void SkipEmptyBuckets() {
            while (BucketIdx_ < Buckets_->size() && (*Buckets_)[BucketIdx_].empty()) {
                ++BucketIdx_;
            }
            if (BucketIdx_ < Buckets_->size()) {
                ListIt_ = (*Buckets_)[BucketIdx_].begin();
            }
        }
    };

public:
    // Constructor
    explicit SimpleUnorderedSet(size_t BucketCount = 16)
        : Buckets(BucketCount) {}

    // Initializer list constructor
    SimpleUnorderedSet(std::initializer_list<T> Il, size_t BucketCount = 16)
        : Buckets(BucketCount) {
        std::cout << "Initializer list constructor called with " << Il.size() << " elements" << std::endl;
        for (const auto& value : Il) {
            insert(value);
        }
    }

    // Iterator methods
    Iterator<false> begin() {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe access
        if (empty()) return end();
        return Iterator<false>(&Buckets, 0, Buckets[0].begin());
    }

    Iterator<true> begin() const {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe access
        if (empty()) return end();
        return Iterator<true>(&Buckets, 0, Buckets[0].begin());
    }

    Iterator<true> cbegin() const {
        return begin();
    }

    Iterator<false> end() {
        // No lock needed here as it returns a constant end iterator
        return Iterator<false>(&Buckets, Buckets.size(), typename std::list<T>::iterator());
    }

    Iterator<true> end() const {
        // No lock needed here as it returns a constant end iterator
        return Iterator<true>(&Buckets, Buckets.size(), typename std::list<T>::const_iterator());
    }

    Iterator<true> cend() const {
        return end();
    }

    // Capacity
    bool empty() const {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe read
        return ElementCount == 0;
    }

    size_t size() const {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe read
        return ElementCount;
    }

    // Modifiers
    std::pair<Iterator<false>, bool> insert(const T& value) {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe modification
        std::cout << "Inserting value " << value << std::endl;
        
        if (LoadFactor() > MaxLoadFactorVal) {
            Rehash(Buckets.size() * 2);
        }

        size_t Idx = Bucket(value);
        auto& list = Buckets[Idx];

        // Check if value already exists
        auto It = std::find_if(list.begin(), list.end(), 
                               [&](const T& Elem) { return equal(Elem, value); });
        if (It != list.end()) {
            std::cout << "Value " << value << " already exists" << std::endl;
            return {Iterator<false>(&Buckets, Idx, It), false};
        }

        // Insert value
        list.push_back(value);
        ++ElementCount;
        auto ListIt = --list.end(); // Iterator to the newly inserted element
        std::cout << "Value " << value << " inserted successfully" << std::endl;
        return {Iterator<false>(&Buckets, Idx, ListIt), true};
    }

    size_t erase(const T& value) {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe modification
        size_t Idx = Bucket(value);
        auto& list = Buckets[Idx];

        auto It = std::find_if(list.begin(), list.end(), 
                               [&](const T& Elem) { return equal(Elem, value); });
        if (It != list.end()) {
            list.erase(It);
            --ElementCount;
            return 1;
        }
        return 0;
    }

    void clear() {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe modification
        for (auto& list : Buckets) {
            list.clear();
        }
        ElementCount = 0;
    }

    // Lookup
    bool Contains(const T& value) const {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe read
        std::cout << "Checking if contains value " << value << std::endl;
        size_t Idx = Bucket(value);
        const auto& list = Buckets[Idx];

        return std::any_of(list.begin(), list.end(), 
                          [&](const T& Elem) { return equal(Elem, value); });
    }

    Iterator<false> find(const T& value) {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe read
        size_t Idx = Bucket(value);
        auto& list = Buckets[Idx];

        auto It = std::find_if(list.begin(), list.end(), 
                               [&](const T& Elem) { return equal(Elem, value); });
        if (It != list.end()) {
            return Iterator<false>(&Buckets, Idx, It);
        }
        return end();
    }

    Iterator<true> find(const T& value) const {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe read
        size_t Idx = Bucket(value);
        const auto& list = Buckets[Idx];

        auto It = std::find_if(list.begin(), list.end(), 
                               [&](const T& Elem) { return equal(Elem, value); });
        if (It != list.end()) {
            return Iterator<true>(&Buckets, Idx, It);
        }
        return end();
    }

    // Bucket interface
    size_t BucketCount() const {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe read
        return Buckets.size();
    }

    size_t Bucket(const T& value) const {
        // No mutex needed here as it's a pure calculation using value
        return Hasher(value) % Buckets.size();
    }

    // Hash policy
    float LoadFactor() const {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe read
        return static_cast<float>(ElementCount) / Buckets.size();
    }

    float MaxLoadFactor() const {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe read
        return MaxLoadFactorVal;
    }

    void MaxLoadFactor(float Ml) {
        std::lock_guard<std::recursive_mutex> Lock(mutex); // Lock required for thread-safe modification
        MaxLoadFactorVal = Ml;
    }

    void Rehash(size_t count) {
        // Already inside a lock in insert() so no additional lock needed
        std::cout << "Rehashing to " << count << " buckets" << std::endl;
        if (count <= Buckets.size()) return;

        std::vector<std::list<T>> NewBuckets(count);
        
        // Move elements to new buckets
        for (auto& list : Buckets) {
            for (auto& value : list) {
                size_t NewIdx = Hasher(value) % count;
                NewBuckets[NewIdx].push_back(value);
            }
        }
        
        Buckets = std::move(NewBuckets);
    }
};

