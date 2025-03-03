/**
 * @file BinaryIndexedTree.hpp
 * @brief Binary Indexed Tree (Fenwick Tree) for prefix sum queries
 * @details Supports O(log n) point updates and prefix/range sum queries.
 *          Uses 1-based indexing internally.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * @class BinaryIndexedTree
 * @brief Fenwick Tree for prefix sums with point updates.
 *
 * A Binary Indexed Tree (BIT) provides efficient:
 *   - Point update: add a value to element at index i
 *   - Prefix query: sum of elements from index 1 to i
 *   - Range query:  sum of elements from index l to r
 *
 * Complexity:
 *   Update     : O(log n)
 *   PrefixSum  : O(log n)
 *   RangeSum   : O(log n)
 *   Build      : O(n log n)
 *   Space      : O(n)
 */
class BinaryIndexedTree {
public:
    /**
     * @brief Construct a BIT of size @p n (1-indexed: elements 1..n).
     * @throws std::invalid_argument if n is 0.
     */
    explicit BinaryIndexedTree(std::size_t n) : size_(n), tree_(n + 1, 0) {
        if (n == 0) {
            throw std::invalid_argument("BinaryIndexedTree: size must be > 0");
        }
    }

    /**
     * @brief Construct a BIT from initial values (0-indexed input, converted to 1-indexed).
     */
    BinaryIndexedTree(const std::vector<long long>& values)
        : size_(values.size()), tree_(values.size() + 1, 0) {
        if (values.empty()) {
            throw std::invalid_argument("BinaryIndexedTree: values must not be empty");
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            Update(i + 1, values[i]);
        }
    }

    /**
     * @brief Add @p delta to element at 1-based @p index.
     * @throws std::out_of_range if index is out of bounds.
     */
    void Update(std::size_t index, long long delta) {
        ValidateIndex(index);
        for (std::size_t i = index; i <= size_; i += (i & (~i + 1))) {
            tree_[i] += delta;
        }
    }

    /**
     * @brief Return sum of elements from 1 to @p index (inclusive).
     * @throws std::out_of_range if index is out of bounds.
     */
    long long PrefixSum(std::size_t index) const {
        ValidateIndex(index);
        long long result = 0;
        for (std::size_t i = index; i > 0; i -= (i & (~i + 1))) {
            result += tree_[i];
        }
        return result;
    }

    /**
     * @brief Return sum of elements from @p left to @p right (inclusive, 1-based).
     * @throws std::out_of_range if indices are out of bounds.
     */
    long long RangeSum(std::size_t left, std::size_t right) const {
        if (left > right) {
            throw std::invalid_argument("BinaryIndexedTree: left > right");
        }
        ValidateIndex(right);
        if (left == 1) return PrefixSum(right);
        ValidateIndex(left);
        return PrefixSum(right) - PrefixSum(left - 1);
    }

    /** @brief Return the number of elements. */
    [[nodiscard]] std::size_t Size() const noexcept { return size_; }

private:
    std::size_t size_;
    std::vector<long long> tree_;

    void ValidateIndex(std::size_t index) const {
        if (index == 0 || index > size_) {
            throw std::out_of_range("BinaryIndexedTree: index out of range");
        }
    }
};
