/**
 * @file BSTNode.hpp
 * @brief Node structure for Binary Search Tree Map (declarations).
 *
 * Each node stores a key-value pair and pointers to left/right children
 * plus a raw parent pointer for bidirectional iterator traversal.
 * Uses std::unique_ptr for automatic memory management.
 */

#pragma once

#include <memory>
#include <utility>

/**
 * @brief Node for the Binary Search Tree Map.
 *
 * @tparam K Key type
 * @tparam V Value type
 */
template <typename K, typename V>
struct BSTNode {
    std::pair<K, V> data;              ///< Key-value pair
    std::unique_ptr<BSTNode> left;     ///< Left child (owned)
    std::unique_ptr<BSTNode> right;    ///< Right child (owned)
    BSTNode* parent{nullptr};          ///< Parent node (non-owning)

    BSTNode(const K& key, const V& value, BSTNode* parent = nullptr);
    BSTNode(K&& key, V&& value, BSTNode* parent = nullptr);
    BSTNode(const std::pair<K, V>& pair, BSTNode* parent = nullptr);
    BSTNode(std::pair<K, V>&& pair, BSTNode* parent = nullptr);
};
