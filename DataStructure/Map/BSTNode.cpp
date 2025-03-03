/**
 * @file BSTNode.cpp
 * @brief Implementation of BSTNode constructors.
 */

#include "BSTNode.hpp"

#include <string>

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

template <typename K, typename V>
BSTNode<K, V>::BSTNode(const K& key, const V& value, BSTNode* parent)
    : data(key, value), left(nullptr), right(nullptr), parent(parent) {}

template <typename K, typename V>
BSTNode<K, V>::BSTNode(K&& key, V&& value, BSTNode* parent)
    : data(std::move(key), std::move(value)),
      left(nullptr), right(nullptr), parent(parent) {}

template <typename K, typename V>
BSTNode<K, V>::BSTNode(const std::pair<K, V>& pair, BSTNode* parent)
    : data(pair), left(nullptr), right(nullptr), parent(parent) {}

template <typename K, typename V>
BSTNode<K, V>::BSTNode(std::pair<K, V>&& pair, BSTNode* parent)
    : data(std::move(pair)), left(nullptr), right(nullptr), parent(parent) {}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template struct BSTNode<int, int>;
template struct BSTNode<int, std::string>;
template struct BSTNode<std::string, int>;
template struct BSTNode<std::string, std::string>;
