#include <iostream>
#include <functional>
#include <mutex>
#include <optional>
#include <memory>
#include <vector>
#include <string>
#include <gtest/gtest.h>

/**
 * @brief A map implementation using a Binary Search Tree.
 * 
 * This class implements a map data structure using a Binary Search Tree (BST).
 * It provides operations for insertion, retrieval, deletion, and traversal.
 * The implementation is thread-safe with mutex locking for concurrent access.
 * 
 * Time complexity:
 * - Average case: O(log n) for insert, find, remove operations
 * - Worst case: O(n) if the tree becomes unbalanced
 * 
 * Space complexity:
 * - O(n) for storing n key-value pairs
 * 
 * @tparam K The key type, must be comparable
 * @tparam V The value type
 * @tparam Compare Comparison function object type, defaults to std::less<K>
 */
template <typename K, typename V, typename Compare = std::less<K>>
class BSTMap {
private:
    /**
     * @brief Node structure for the Binary Search Tree.
     */
    struct Node {
        std::pair<K, V> data; // Key-value pair
        std::unique_ptr<Node> left;  // Left child
        std::unique_ptr<Node> right; // Right child
        
        /**
         * @brief Construct a new Node with the given key and value.
         * 
         * @param key The key to store
         * @param value The value to store
         */
        Node(const K& key, const V& value) 
            : data(key, value), left(nullptr), right(nullptr) {}
    };
    
    std::unique_ptr<Node> root; // Root of the BST
    Compare comp;              // Comparison function object
    mutable std::mutex mutex;  // Mutex for thread safety - mutable to allow locking in const methods
    size_t size_;              // Number of elements in the map
    
public:
    /**
     * @brief Construct a new empty BSTMap.
     */
    BSTMap() : root(nullptr), size_(0) {}
    
    /**
     * @brief Destructor that cleans up the tree.
     * 
     * The unique_ptr's will automatically delete the nodes,
     * but we explicitly clear the tree for clarity.
     */
    ~BSTMap() {
        clear();
    }
    
    /**
     * @brief Insert a key-value pair into the map.
     * 
     * If the key already exists, its value is updated.
     * 
     * Time complexity:
     * - Average case: O(log n)
     * - Worst case: O(n) for an unbalanced tree
     * 
     * @param key The key to insert
     * @param value The value to associate with the key
     * @return true if a new key-value pair was inserted, false if the key already existed
     */
    bool insert(const K& key, const V& value) {
        std::lock_guard<std::mutex> lock(mutex); // Lock during insertion for thread safety
        return insertImpl(root, key, value);
    }
    
    /**
     * @brief Find a value by key.
     * 
     * Time complexity:
     * - Average case: O(log n)
     * - Worst case: O(n) for an unbalanced tree
     * 
     * @param key The key to find
     * @return std::optional<V> containing the value if found, or std::nullopt if not found
     */
    std::optional<V> find(const K& key) const {
        std::lock_guard<std::mutex> lock(mutex); // Lock during search for thread safety
        return findImpl(root, key);
    }
    
    /**
     * @brief Remove a key-value pair from the map.
     *
     * Time complexity:
     * - Average case: O(log n)
     * - Worst case: O(n) for an unbalanced tree
     * 
     * @param key The key to remove
     * @return true if the key was found and removed, false otherwise
     */
    bool remove(const K& key) {
        std::lock_guard<std::mutex> lock(mutex); // Lock during removal for thread safety
        return removeImpl(root, key);
    }
    
    /**
     * @brief Get the size of the map.
     * 
     * Time complexity: O(1)
     * 
     * @return The number of key-value pairs in the map
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock for thread safety
        return size_;
    }
    
    /**
     * @brief Check if the map is empty.
     * 
     * Time complexity: O(1)
     * 
     * @return true if the map is empty, false otherwise
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock for thread safety
        return size_ == 0;
    }
    
    /**
     * @brief Clear the map, removing all key-value pairs.
     * 
     * Time complexity: O(1) for the operation itself,
     * but memory deallocation is O(n)
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex); // Lock during clearing for thread safety
        root.reset();
        size_ = 0;
    }
    
    /**
     * @brief Get all keys in the map in sorted order.
     * 
     * Time complexity: O(n)
     * 
     * @return A vector containing all keys in sorted order
     */
    std::vector<K> keys() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock for thread safety
        std::vector<K> result;
        inorderKeys(root, result);
        return result;
    }
    
    /**
     * @brief Get all values in the map in the order of sorted keys.
     * 
     * Time complexity: O(n)
     * 
     * @return A vector containing all values in the order of sorted keys
     */
    std::vector<V> values() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock for thread safety
        std::vector<V> result;
        inorderValues(root, result);
        return result;
    }
    
    /**
     * @brief Get all key-value pairs in the map in sorted order by key.
     * 
     * Time complexity: O(n)
     * 
     * @return A vector containing all key-value pairs in sorted order by key
     */
    std::vector<std::pair<K, V>> entries() const {
        std::lock_guard<std::mutex> lock(mutex); // Lock for thread safety
        std::vector<std::pair<K, V>> result;
        inorderEntries(root, result);
        return result;
    }
    
private:
    /**
     * @brief Implementation of insert.
     * 
     * @param node The current node reference
     * @param key The key to insert
     * @param value The value to associate with the key
     * @return true if a new key-value pair was inserted, false if the key already existed
     */
    bool insertImpl(std::unique_ptr<Node>& node, const K& key, const V& value) {
        if (!node) {
            // Create a new node for this key-value pair
            node = std::make_unique<Node>(key, value);
            size_++;
            return true;
        }
        
        // Check for key equality using the comparator
        // Two keys are equal if neither is less than the other
        if (!comp(key, node->data.first) && !comp(node->data.first, key)) {
            // Key already exists, update value
            node->data.second = value;
            return false;
        }
        
        if (comp(key, node->data.first)) {
            // Key is less than current node's key, go left
            return insertImpl(node->left, key, value);
        } else {
            // Key is greater than current node's key, go right
            return insertImpl(node->right, key, value);
        }
    }
    
    /**
     * @brief Implementation of find.
     * 
     * @param node The current node
     * @param key The key to find
     * @return std::optional<V> containing the value if found, or std::nullopt if not found
     */
    std::optional<V> findImpl(const std::unique_ptr<Node>& node, const K& key) const {
        if (!node) {
            // Key not found
            return std::nullopt;
        }
        
        // Check for key equality using the comparator
        // Two keys are equal if neither is less than the other
        if (!comp(key, node->data.first) && !comp(node->data.first, key)) {
            // Key found
            return node->data.second;
        }
        
        if (comp(key, node->data.first)) {
            // Key is less than current node's key, go left
            return findImpl(node->left, key);
        } else {
            // Key is greater than current node's key, go right
            return findImpl(node->right, key);
        }
    }
    
    /**
     * @brief Implementation of remove.
     * 
     * @param node The current node reference
     * @param key The key to remove
     * @return true if the key was found and removed, false otherwise
     */
    bool removeImpl(std::unique_ptr<Node>& node, const K& key) {
        if (!node) {
            // Key not found
            return false;
        }
        
        if (comp(key, node->data.first)) {
            // Key is less than current node's key, go left
            return removeImpl(node->left, key);
        } else if (comp(node->data.first, key)) {
            // Key is greater than current node's key, go right
            return removeImpl(node->right, key);
        } else if (!comp(key, node->data.first) && !comp(node->data.first, key)) {
            // Found the node to remove (keys are equal according to the comparator)
            
            // Case 1: Node with no children
            if (!node->left && !node->right) {
                node.reset();
                size_--;
                return true;
            }
            // Case 2: Node with only one child
            else if (!node->left) {
                node = std::move(node->right);
                size_--;
                return true;
            }
            else if (!node->right) {
                node = std::move(node->left);
                size_--;
                return true;
            }
            // Case 3: Node with two children
            else {
                // Find the inorder successor (smallest node in right subtree)
                Node* successor = findMin(node->right.get());
                
                // Copy successor's data to current node
                node->data = successor->data;
                
                // Remove the successor from the right subtree
                return removeImpl(node->right, successor->data.first);
            }
        }
    }
    
    /**
     * @brief Find the node with the minimum key in the subtree.
     * 
     * @param node The root of the subtree
     * @return Pointer to the node with the minimum key
     */
    Node* findMin(Node* node) const {
        if (!node) return nullptr;
        
        while (node->left) {
            node = node->left.get();
        }
        return node;
    }
    
    /**
     * @brief Collect keys in inorder traversal.
     * 
     * @param node The current node
     * @param result Vector to store the keys
     */
    void inorderKeys(const std::unique_ptr<Node>& node, std::vector<K>& result) const {
        if (node) {
            inorderKeys(node->left, result);
            result.push_back(node->data.first);
            inorderKeys(node->right, result);
        }
    }
    
    /**
     * @brief Collect values in inorder traversal.
     * 
     * @param node The current node
     * @param result Vector to store the values
     */
    void inorderValues(const std::unique_ptr<Node>& node, std::vector<V>& result) const {
        if (node) {
            inorderValues(node->left, result);
            result.push_back(node->data.second);
            inorderValues(node->right, result);
        }
    }
    
    /**
     * @brief Collect key-value pairs in inorder traversal.
     * 
     * @param node The current node
     * @param result Vector to store the key-value pairs
     */
    void inorderEntries(const std::unique_ptr<Node>& node, std::vector<std::pair<K, V>>& result) const {
        if (node) {
            inorderEntries(node->left, result);
            result.push_back(node->data);
            inorderEntries(node->right, result);
        }
    }
};

// Google Test cases
class BSTMapIntStringTest : public ::testing::Test {
protected:
    BSTMap<int, std::string> map;
    
    void SetUp() override {
        // Set up a fresh map for each test
        map.clear();
    }
};

// Test insertion and finding
TEST_F(BSTMapIntStringTest, InsertAndFind) {
    // Test insertion
    EXPECT_TRUE(map.insert(1, "one"));
    EXPECT_TRUE(map.insert(2, "two"));
    EXPECT_TRUE(map.insert(3, "three"));
    
    // Test finding existing keys
    auto value1 = map.find(1);
    EXPECT_TRUE(value1.has_value());
    EXPECT_EQ(*value1, "one");
    
    auto value2 = map.find(2);
    EXPECT_TRUE(value2.has_value());
    EXPECT_EQ(*value2, "two");
    
    auto value3 = map.find(3);
    EXPECT_TRUE(value3.has_value());
    EXPECT_EQ(*value3, "three");
    
    // Test finding non-existent key
    auto value4 = map.find(4);
    EXPECT_FALSE(value4.has_value());
}

// Test updating values
TEST_F(BSTMapIntStringTest, UpdateValue) {
    // Insert initial value
    EXPECT_TRUE(map.insert(1, "one"));
    
    // Update value
    EXPECT_FALSE(map.insert(1, "ONE"));
    
    // Verify updated value
    auto value = map.find(1);
    EXPECT_TRUE(value.has_value());
    EXPECT_EQ(*value, "ONE");
}

// Test removing keys
TEST_F(BSTMapIntStringTest, RemoveKey) {
    // Insert values
    map.insert(2, "two");
    map.insert(1, "one");
    map.insert(3, "three");
    
    // Remove a leaf node
    EXPECT_TRUE(map.remove(1));
    EXPECT_FALSE(map.find(1).has_value());
    
    // Remove a node with one child
    map.insert(4, "four");
    EXPECT_TRUE(map.remove(3));
    EXPECT_FALSE(map.find(3).has_value());
    
    // Remove a node with two children
    map.insert(1, "one");
    map.insert(3, "three");
    EXPECT_TRUE(map.remove(2));
    EXPECT_FALSE(map.find(2).has_value());
    
    // Remove non-existent key
    EXPECT_FALSE(map.remove(5));
}

// Test size and empty
TEST_F(BSTMapIntStringTest, SizeAndEmpty) {
    // Test empty map
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
    
    // Test after insertion
    map.insert(1, "one");
    EXPECT_FALSE(map.empty());
    EXPECT_EQ(map.size(), 1);
    
    map.insert(2, "two");
    EXPECT_EQ(map.size(), 2);
    
    // Test after removal
    map.remove(1);
    EXPECT_EQ(map.size(), 1);
    
    map.remove(2);
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
}

// Test clearing the map
TEST_F(BSTMapIntStringTest, Clear) {
    // Insert values
    map.insert(1, "one");
    map.insert(2, "two");
    map.insert(3, "three");
    
    // Clear the map
    map.clear();
    
    // Test after clearing
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
    EXPECT_FALSE(map.find(1).has_value());
    EXPECT_FALSE(map.find(2).has_value());
    EXPECT_FALSE(map.find(3).has_value());
}

// Test keys and values
TEST_F(BSTMapIntStringTest, KeysAndValues) {
    // Insert values in arbitrary order
    map.insert(3, "three");
    map.insert(1, "one");
    map.insert(2, "two");
    
    // Test keys
    std::vector<int> expectedKeys = {1, 2, 3};
    EXPECT_EQ(map.keys(), expectedKeys);
    
    // Test values (order matches sorted keys)
    std::vector<std::string> expectedValues = {"one", "two", "three"};
    EXPECT_EQ(map.values(), expectedValues);
    
    // Test entries
    std::vector<std::pair<int, std::string>> expectedEntries = {
        {1, "one"}, {2, "two"}, {3, "three"}
    };
    EXPECT_EQ(map.entries(), expectedEntries);
}

// Test edge case: large number of insertions
TEST_F(BSTMapIntStringTest, LargeNumberOfInsertions) {
    const int N = 1000;
    
    // Insert a large number of key-value pairs
    for (int i = 0; i < N; i++) {
        EXPECT_TRUE(map.insert(i, std::to_string(i)));
    }
    
    // Verify size
    EXPECT_EQ(map.size(), N);
    
    // Verify all values can be found
    for (int i = 0; i < N; i++) {
        auto value = map.find(i);
        EXPECT_TRUE(value.has_value());
        EXPECT_EQ(*value, std::to_string(i));
    }
}

// Test with different data types
class BSTMapStringDoubleTest : public ::testing::Test {
protected:
    BSTMap<std::string, double> map;
    
    void SetUp() override {
        map.clear();
    }
};

TEST_F(BSTMapStringDoubleTest, BasicOperations) {
    // Test with string keys and double values
    EXPECT_TRUE(map.insert("pi", 3.14159));
    EXPECT_TRUE(map.insert("e", 2.71828));
    EXPECT_TRUE(map.insert("sqrt2", 1.41421));
    
    // Test finding
    auto pi = map.find("pi");
    EXPECT_TRUE(pi.has_value());
    EXPECT_DOUBLE_EQ(*pi, 3.14159);
    
    // Test updating
    EXPECT_FALSE(map.insert("pi", 3.14));
    pi = map.find("pi");
    EXPECT_DOUBLE_EQ(*pi, 3.14);
    
    // Test size
    EXPECT_EQ(map.size(), 3);
    
    // Test removal
    EXPECT_TRUE(map.remove("e"));
    EXPECT_EQ(map.size(), 2);
    EXPECT_FALSE(map.find("e").has_value());
}

// Test custom comparator
struct CaseInsensitiveCompare {
    bool operator()(const std::string& a, const std::string& b) const {
        return std::lexicographical_compare(
            a.begin(), a.end(), b.begin(), b.end(),
            [](char c1, char c2) { return std::tolower(c1) < std::tolower(c2); }
        );
    }
};

class BSTMapCustomCompareTest : public ::testing::Test {
protected:
    BSTMap<std::string, int, CaseInsensitiveCompare> map;
    
    void SetUp() override {
        map.clear();
    }
};

TEST_F(BSTMapCustomCompareTest, CaseInsensitiveKeys) {
    // Insert with lowercase keys
    EXPECT_TRUE(map.insert("apple", 1));
    EXPECT_TRUE(map.insert("banana", 2));
    
    // Find with different case
    auto value1 = map.find("APPLE");
    EXPECT_TRUE(value1.has_value());
    EXPECT_EQ(*value1, 1);
    
    auto value2 = map.find("Banana");
    EXPECT_TRUE(value2.has_value());
    EXPECT_EQ(*value2, 2);
    
    // Update with different case
    EXPECT_FALSE(map.insert("APPLE", 10));
    value1 = map.find("apple");
    EXPECT_EQ(*value1, 10);
    
    // Remove with different case
    EXPECT_TRUE(map.remove("bAnAnA"));
    EXPECT_FALSE(map.find("banana").has_value());
}

// Test thread safety
TEST(BSTMapThreadSafetyTest, ConcurrentOperations) {
    BSTMap<int, int> map;
    
    // This is a basic test - a more thorough test would use actual threads
    // and synchronization mechanisms
    
    // Insert some values
    for (int i = 0; i < 100; i++) {
        map.insert(i, i * 10);
    }
    
    // Verify all values
    for (int i = 0; i < 100; i++) {
        auto value = map.find(i);
        EXPECT_TRUE(value.has_value());
        EXPECT_EQ(*value, i * 10);
    }
}

/**
 * Alternative implementations for a map data structure with trade-offs:
 * 
 * 1. Balanced BST (AVL Tree or Red-Black Tree):
 *    - Pros: Guarantees O(log n) time complexity for all operations, even in worst case
 *    - Cons: Higher implementation complexity, additional memory overhead for balance information
 * 
 * 2. Hash Table:
 *    - Pros: O(1) average time complexity for insert, find, remove operations
 *    - Cons: Does not maintain key order, requires good hash function, potential for hash collisions
 * 
 * 3. Skip List:
 *    - Pros: Probabilistic O(log n) time complexity, simpler implementation than balanced trees
 *    - Cons: Uses more memory, performance guarantees are probabilistic
 * 
 * 4. B-Tree:
 *    - Pros: Better cache locality, good for disk-based storage
 *    - Cons: More complex implementation, higher memory overhead
 * 
 * Our current BST implementation has these trade-offs:
 * - Pros: Simple implementation, efficient for mostly random insertions
 * - Cons: Can degrade to O(n) time complexity if insertions occur in sorted order
 *         (tree becomes unbalanced)
 * 
 * Security considerations:
 * - The current implementation is vulnerable to algorithmic complexity attacks 
 *   if an attacker can control the order of keys (e.g., inserting in sorted order)
 * - For production use, a balanced tree implementation would be more secure
 */

// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}