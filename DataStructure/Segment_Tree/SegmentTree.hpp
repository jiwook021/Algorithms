/**
 * @file SegmentTree.hpp
 * @brief Generic Segment Tree for efficient range queries and point updates.
 *
 * @details
 * A segment tree stores an array of N elements and supports two operations
 * in O(log N) time each:
 *   - Point update:  change one element.
 *   - Range query:   combine all elements in a contiguous range [L, R]
 *                    using an associative binary operation (sum, min, max,
 *                    gcd, ...).
 *
 * Uses a 1-indexed implicit binary tree stored in a std::vector.
 * Leaves hold original data; internal nodes store aggregated results.
 *
 * The operation is passed at construction time as a std::function, so a
 * single template parameter (the value type T) is all that is needed.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

/**
 * @class SegmentTree
 * @brief Segment Tree with O(log N) point update and range query.
 *
 * Memory layout (example: n=4 elements, size_=4):
 *
 *            tree_[1]              <- root: aggregate of ALL elements
 *           /         \
 *      tree_[2]     tree_[3]      <- internal nodes: partial aggregates
 *      /    \        /    \
 *   tree_[4] tree_[5] tree_[6] tree_[7]  <- leaves: actual data values
 *    (idx 0)  (idx 1)  (idx 2)  (idx 3)
 *
 * Leaves live at positions [size_ .. 2*size_-1].  Each parent at position i
 * stores op_(child_left, child_right).
 *
 * @tparam T  Element type (int, double, long long, ...).
 */
template <typename T>
class SegmentTree {
public:
    /**
     * Construct and build the tree from a data vector.
     * @param data      Source values (0-indexed).
     * @param identity  Identity element for the operation
     *                  (0 for sum, INT_MAX for min, INT_MIN for max, ...).
     * @param op        Associative binary operation (default: addition).
     */
    explicit SegmentTree(const std::vector<T>& data,
                         T identity = T{},
                         std::function<T(T, T)> op = std::plus<T>{});

    /**
     * Point update: set element at 0-based index to a new value.
     * Propagates the change from leaf to root.  O(log N).
     */
    void Update(std::size_t index, T value);

    /**
     * Range query over [left, right] (0-based, inclusive on both ends).
     * Returns the aggregate of all elements in the range.  O(log N).
     */
    [[nodiscard]] T Query(std::size_t left, std::size_t right) const;

    /** Number of logical elements the tree was built with. */
    [[nodiscard]] std::size_t Size() const;

private:
    /**
     * Build the tree bottom-up from a data vector.
     * Rounds size_ to the next power of 2, copies data into leaves,
     * then fills internal nodes via op_.
     */
    void Build(const std::vector<T>& data);

    /**
     * Recursive range-query helper.
     * Three cases per node:
     *   1. Completely outside query range -> return identity_.
     *   2. Completely inside query range  -> return tree_[node].
     *   3. Partial overlap -> recurse into both children and combine.
     */
    T QueryHelper(std::size_t node, std::size_t start, std::size_t end,
                  std::size_t left, std::size_t right) const;

    std::vector<T> tree_;
    std::size_t size_;
    T identity_;
    std::function<T(T, T)> op_;
};
