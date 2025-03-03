/**
 * @file ThreadedBinaryTree.hpp
 * @brief Threaded Binary Search Tree for in-order traversal without recursion or stack.
 * @details In a threaded binary tree, null left/right pointers are replaced with
 *          "threads" pointing to the in-order predecessor/successor. This enables
 *          O(n) in-order traversal using only O(1) auxiliary space.
 *
 * Thread types:
 *   - Right thread: If a node has no right child, its right pointer points to
 *     its in-order successor.
 *   - Left thread: If a node has no left child, its left pointer points to
 *     its in-order predecessor.
 *
 * Complexity:
 *   Insert      : O(h) where h = tree height
 *   Search      : O(h)
 *   In-order    : O(n) with O(1) space
 *   Space       : O(n)
 */

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

/**
 * @class ThreadedBinaryTree
 * @brief Threaded BST supporting insert, search, and stack-free in-order traversal.
 */
class ThreadedBinaryTree {
public:
    /** @brief Construct an empty threaded binary tree. */
    ThreadedBinaryTree() noexcept : root_(nullptr), size_(0) {}

    /** @brief Destructor frees all nodes. */
    ~ThreadedBinaryTree() noexcept {
        Clear();
    }

    /// Non-copyable (raw pointers with threads).
    ThreadedBinaryTree(const ThreadedBinaryTree&) = delete;
    ThreadedBinaryTree& operator=(const ThreadedBinaryTree&) = delete;

    /// Move constructor.
    ThreadedBinaryTree(ThreadedBinaryTree&& other) noexcept
        : root_(other.root_), size_(other.size_) {
        other.root_ = nullptr;
        other.size_ = 0;
    }

    /// Move assignment.
    ThreadedBinaryTree& operator=(ThreadedBinaryTree&& other) noexcept {
        if (this != &other) {
            Clear();
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Insert a value into the tree.
     * @param value The value to insert (duplicates are ignored).
     * @return true if inserted, false if duplicate.
     */
    bool Insert(int value) {
        if (!root_) {
            root_ = new Node(value);
            ++size_;
            return true;
        }

        Node* cur = root_;
        Node* parent = nullptr;

        while (cur) {
            if (value == cur->data) {
                return false;  // duplicate
            }
            parent = cur;
            if (value < cur->data) {
                if (!cur->leftThread) {
                    cur = cur->left;
                } else {
                    break;
                }
            } else {
                if (!cur->rightThread) {
                    cur = cur->right;
                } else {
                    break;
                }
            }
        }

        Node* newNode = new Node(value);

        if (value < parent->data) {
            // New node's left thread -> parent's left thread target (predecessor)
            newNode->left = parent->left;
            newNode->leftThread = true;
            // New node's right thread -> parent (successor)
            newNode->right = parent;
            newNode->rightThread = true;
            // Parent's left now points to real child
            parent->left = newNode;
            parent->leftThread = false;
        } else {
            // New node's right thread -> parent's right thread target (successor)
            newNode->right = parent->right;
            newNode->rightThread = true;
            // New node's left thread -> parent (predecessor)
            newNode->left = parent;
            newNode->leftThread = true;
            // Parent's right now points to real child
            parent->right = newNode;
            parent->rightThread = false;
        }

        ++size_;
        return true;
    }

    /**
     * @brief Search for a value in the tree.
     * @param value The value to search for.
     * @return true if found.
     */
    bool Search(int value) const noexcept {
        Node* cur = root_;
        while (cur) {
            if (value == cur->data) return true;
            if (value < cur->data) {
                if (cur->leftThread) return false;
                cur = cur->left;
            } else {
                if (cur->rightThread) return false;
                cur = cur->right;
            }
        }
        return false;
    }

    /**
     * @brief Return the in-order traversal as a vector.
     *
     * Uses threads to traverse without recursion or a stack.
     * Time: O(n), Space: O(1) auxiliary (O(n) for the result vector).
     */
    std::vector<int> Inorder() const {
        std::vector<int> result;
        result.reserve(size_);

        if (!root_) return result;

        // Find the leftmost node
        Node* cur = root_;
        while (cur && !cur->leftThread && cur->left) {
            cur = cur->left;
        }

        // Follow threads/right pointers for in-order traversal
        while (cur) {
            result.push_back(cur->data);
            cur = InorderSuccessor(cur);
        }

        return result;
    }

    /** @brief Remove all nodes. */
    void Clear() noexcept {
        if (!root_) return;

        // Collect all nodes via in-order, then delete
        // (cannot just follow threads and delete, as it breaks threads)
        std::vector<Node*> nodes;
        nodes.reserve(size_);
        CollectNodes(root_, nodes);
        for (Node* n : nodes) {
            delete n;
        }
        root_ = nullptr;
        size_ = 0;
    }

    /** @brief Check if the tree is empty. */
    [[nodiscard]] bool Empty() const noexcept { return size_ == 0; }

    /** @brief Return the number of elements. */
    [[nodiscard]] std::size_t Size() const noexcept { return size_; }

private:
    /** @brief Internal node with thread flags. */
    struct Node {
        int data;
        Node* left;
        Node* right;
        bool leftThread;   ///< true if left is a thread (not a real child)
        bool rightThread;  ///< true if right is a thread (not a real child)

        explicit Node(int val)
            : data(val), left(nullptr), right(nullptr),
              leftThread(true), rightThread(true) {}
    };

    Node* root_;
    std::size_t size_;

    /**
     * @brief Find the in-order successor of a node.
     * @param node Current node.
     * @return Pointer to the in-order successor, or nullptr if none.
     */
    static Node* InorderSuccessor(Node* node) noexcept {
        if (node->rightThread) {
            return node->right;  // thread points to successor
        }
        // Real right child: go right once, then as far left as possible
        Node* cur = node->right;
        while (cur && !cur->leftThread && cur->left) {
            cur = cur->left;
        }
        return cur;
    }

    /** @brief Collect all reachable nodes (for deletion). */
    void CollectNodes(Node* node, std::vector<Node*>& out) const {
        if (!node) return;
        // Go left if real child
        if (!node->leftThread && node->left) {
            CollectNodes(node->left, out);
        }
        out.push_back(node);
        // Go right if real child
        if (!node->rightThread && node->right) {
            CollectNodes(node->right, out);
        }
    }
};
