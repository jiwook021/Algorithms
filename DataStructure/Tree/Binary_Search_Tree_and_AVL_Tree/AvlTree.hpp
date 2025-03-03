/**
 * @file AvlTree.hpp
 * @brief AVL-balanced Binary Search Tree implementation.
 * @details A self-balancing BST where the height difference between left and
 *          right subtrees of any node is at most 1. Balancing is achieved through
 *          four rotation types: LL, RR, LR, and RL.
 *
 * Complexity:
 *   Insert  : O(log n)
 *   Search  : O(log n)
 *   Remove  : O(log n)
 *   Traversal: O(n)
 *   Space   : O(n)
 */

#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

/**
 * @class AvlTree
 * @brief Self-balancing AVL binary search tree.
 * @tparam T Element type (must be comparable with <, ==).
 */
template <typename T>
class AvlTree {
private:
    /** @brief Internal node structure. */
    struct Node {
        T data;
        Node* left;
        Node* right;
        int height;

        explicit Node(const T& val)
            : data(val), left(nullptr), right(nullptr), height(1) {}
    };

    Node* root_;
    std::size_t size_;

    // ── Height helpers ──────────────────────────────────────────────

    /** @brief Return height of a node (0 for nullptr). */
    static int Height(const Node* node) noexcept {
        return node ? node->height : 0;
    }

    /** @brief Recompute height from children. */
    static void UpdateHeight(Node* node) noexcept {
        if (node) {
            int lh = Height(node->left);
            int rh = Height(node->right);
            node->height = 1 + (lh > rh ? lh : rh);
        }
    }

    /** @brief Balance factor = height(left) - height(right). */
    static int BalanceFactor(const Node* node) noexcept {
        return node ? Height(node->left) - Height(node->right) : 0;
    }

    // ── Rotations ───────────────────────────────────────────────────

    /**
     * @brief Right rotation (LL case).
     * @param y The unbalanced node.
     * @return New root of the subtree.
     */
    Node* RotateRight(Node* y) noexcept {
        Node* x = y->left;
        Node* t2 = x->right;
        x->right = y;
        y->left = t2;
        UpdateHeight(y);
        UpdateHeight(x);
        return x;
    }

    /**
     * @brief Left rotation (RR case).
     * @param x The unbalanced node.
     * @return New root of the subtree.
     */
    Node* RotateLeft(Node* x) noexcept {
        Node* y = x->right;
        Node* t2 = y->left;
        y->left = x;
        x->right = t2;
        UpdateHeight(x);
        UpdateHeight(y);
        return y;
    }

    /**
     * @brief Rebalance a node after insertion or deletion.
     * @param node The node to check and potentially rotate.
     * @return The new root of the subtree.
     */
    Node* Balance(Node* node) noexcept {
        UpdateHeight(node);
        int bf = BalanceFactor(node);

        // Left-heavy
        if (bf > 1) {
            if (BalanceFactor(node->left) < 0) {
                node->left = RotateLeft(node->left);  // LR case
            }
            return RotateRight(node);  // LL case
        }

        // Right-heavy
        if (bf < -1) {
            if (BalanceFactor(node->right) > 0) {
                node->right = RotateRight(node->right);  // RL case
            }
            return RotateLeft(node);  // RR case
        }

        return node;
    }

    // ── Recursive operations ────────────────────────────────────────

    /** @brief Recursive insert. */
    Node* InsertImpl(Node* node, const T& value) {
        if (!node) {
            ++size_;
            return new Node(value);
        }
        if (value < node->data) {
            node->left = InsertImpl(node->left, value);
        } else if (node->data < value) {
            node->right = InsertImpl(node->right, value);
        } else {
            return node;  // duplicate, no insert
        }
        return Balance(node);
    }

    /** @brief Find minimum node in subtree. */
    static Node* FindMin(Node* node) noexcept {
        while (node && node->left) {
            node = node->left;
        }
        return node;
    }

    /** @brief Recursive remove. */
    Node* RemoveImpl(Node* node, const T& value) {
        if (!node) return nullptr;

        if (value < node->data) {
            node->left = RemoveImpl(node->left, value);
        } else if (node->data < value) {
            node->right = RemoveImpl(node->right, value);
        } else {
            // Found the node to remove
            if (!node->left || !node->right) {
                Node* child = node->left ? node->left : node->right;
                delete node;
                --size_;
                return child;
            }
            // Two children: replace with in-order successor
            Node* successor = FindMin(node->right);
            node->data = successor->data;
            node->right = RemoveImpl(node->right, successor->data);
        }
        return Balance(node);
    }

    /** @brief Recursive search. */
    Node* SearchImpl(Node* node, const T& value) const noexcept {
        if (!node) return nullptr;
        if (value < node->data) return SearchImpl(node->left, value);
        if (node->data < value) return SearchImpl(node->right, value);
        return node;
    }

    /** @brief Recursive in-order traversal. */
    void InorderImpl(const Node* node, std::vector<T>& result) const {
        if (!node) return;
        InorderImpl(node->left, result);
        result.push_back(node->data);
        InorderImpl(node->right, result);
    }

    /** @brief Recursive pre-order traversal. */
    void PreorderImpl(const Node* node, std::vector<T>& result) const {
        if (!node) return;
        result.push_back(node->data);
        PreorderImpl(node->left, result);
        PreorderImpl(node->right, result);
    }

    /** @brief Recursive post-order traversal. */
    void PostorderImpl(const Node* node, std::vector<T>& result) const {
        if (!node) return;
        PostorderImpl(node->left, result);
        PostorderImpl(node->right, result);
        result.push_back(node->data);
    }

    /** @brief Recursively free all nodes. */
    void DestroyImpl(Node* node) noexcept {
        if (!node) return;
        DestroyImpl(node->left);
        DestroyImpl(node->right);
        delete node;
    }

    /** @brief Deep copy. */
    Node* CopyImpl(const Node* node) {
        if (!node) return nullptr;
        Node* copy = new Node(node->data);
        copy->height = node->height;
        copy->left = CopyImpl(node->left);
        copy->right = CopyImpl(node->right);
        return copy;
    }

public:
    // ── Construction / destruction ──────────────────────────────────

    /** @brief Construct an empty AVL tree. */
    AvlTree() noexcept : root_(nullptr), size_(0) {}

    /** @brief Construct from initializer list. */
    AvlTree(std::initializer_list<T> init) : AvlTree() {
        for (const auto& v : init) {
            Insert(v);
        }
    }

    /** @brief Copy constructor. */
    AvlTree(const AvlTree& other) : root_(nullptr), size_(other.size_) {
        root_ = CopyImpl(other.root_);
    }

    /** @brief Copy assignment. */
    AvlTree& operator=(const AvlTree& other) {
        if (this != &other) {
            DestroyImpl(root_);
            size_ = other.size_;
            root_ = CopyImpl(other.root_);
        }
        return *this;
    }

    /** @brief Move constructor. */
    AvlTree(AvlTree&& other) noexcept
        : root_(other.root_), size_(other.size_) {
        other.root_ = nullptr;
        other.size_ = 0;
    }

    /** @brief Move assignment. */
    AvlTree& operator=(AvlTree&& other) noexcept {
        if (this != &other) {
            DestroyImpl(root_);
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /** @brief Destructor. */
    ~AvlTree() noexcept { DestroyImpl(root_); }

    // ── Public API ──────────────────────────────────────────────────

    /**
     * @brief Insert a value into the tree.
     * @param value The value to insert (duplicates ignored).
     */
    void Insert(const T& value) {
        root_ = InsertImpl(root_, value);
    }

    /**
     * @brief Remove a value from the tree.
     * @param value The value to remove.
     * @return true if the value was found and removed.
     */
    bool Remove(const T& value) {
        std::size_t oldSize = size_;
        root_ = RemoveImpl(root_, value);
        return size_ < oldSize;
    }

    /**
     * @brief Search for a value in the tree.
     * @param value The value to search for.
     * @return true if found.
     */
    bool Search(const T& value) const noexcept {
        return SearchImpl(root_, value) != nullptr;
    }

    /** @brief Return elements in sorted (in-order) order. */
    std::vector<T> Inorder() const {
        std::vector<T> result;
        result.reserve(size_);
        InorderImpl(root_, result);
        return result;
    }

    /** @brief Return elements in pre-order. */
    std::vector<T> Preorder() const {
        std::vector<T> result;
        result.reserve(size_);
        PreorderImpl(root_, result);
        return result;
    }

    /** @brief Return elements in post-order. */
    std::vector<T> Postorder() const {
        std::vector<T> result;
        result.reserve(size_);
        PostorderImpl(root_, result);
        return result;
    }

    /** @brief Remove all elements. */
    void Clear() noexcept {
        DestroyImpl(root_);
        root_ = nullptr;
        size_ = 0;
    }

    /** @brief Check if the tree is empty. */
    [[nodiscard]] bool Empty() const noexcept { return size_ == 0; }

    /** @brief Return the number of elements. */
    [[nodiscard]] std::size_t Size() const noexcept { return size_; }

    /** @brief Return the height of the tree. */
    [[nodiscard]] int TreeHeight() const noexcept { return Height(root_); }
};
