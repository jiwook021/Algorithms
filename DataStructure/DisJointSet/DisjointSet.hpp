/**
 * @file DisjointSet.hpp
 * @brief Disjoint Set (Union-Find) data structure
 * @details Implements a disjoint set forest with union by rank and path compression.
 *          Supports near-O(1) amortized union and find operations via the inverse
 *          Ackermann function bound O(alpha(n)).
 */

#pragma once

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

/**
 * @class DisjointSet
 * @brief Union-Find forest with union by rank and path compression.
 *
 * Each element belongs to exactly one set, identified by a representative (root).
 * Two key optimisations keep operations nearly O(1):
 *   - Path compression: during Find(), point every visited node directly at the root.
 *   - Union by rank: attach the shorter tree under the taller one.
 *
 * Complexity (amortised per operation):
 *   Find   : O(alpha(n))  ~  O(1) in practice
 *   Union  : O(alpha(n))
 *   Space  : O(n)
 */
class DisjointSet {
public:
    /**
     * @brief Construct a disjoint set with @p size elements, each in its own set.
     * @param size Number of elements (indexed 0 .. size-1).
     * @throws std::invalid_argument if size is 0.
     */
    explicit DisjointSet(std::size_t size) : parent_(size), rank_(size, 0) {
        if (size == 0) {
            throw std::invalid_argument("DisjointSet: size must be > 0");
        }
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    /**
     * @brief Find the representative (root) of the set containing @p x.
     * @param x Element index.
     * @return Root index of the set containing @p x.
     * @throws std::out_of_range if x >= Size().
     *
     * Uses path compression: every node on the path to the root is
     * re-parented directly to the root, flattening the tree.
     */
    std::size_t Find(std::size_t x) {
        ValidateIndex(x);
        if (parent_[x] != x) {
            parent_[x] = Find(parent_[x]);
        }
        return parent_[x];
    }

    /**
     * @brief Merge the sets containing @p a and @p b.
     * @return true if a merge occurred, false if already in the same set.
     * @throws std::out_of_range if a or b >= Size().
     *
     * Uses union by rank: the tree with the smaller rank is attached
     * under the tree with the larger rank, keeping the structure shallow.
     */
    bool Union(std::size_t a, std::size_t b) {
        std::size_t rootA = Find(a);
        std::size_t rootB = Find(b);
        if (rootA == rootB) {
            return false;
        }
        if (rank_[rootA] < rank_[rootB]) {
            parent_[rootA] = rootB;
        } else if (rank_[rootA] > rank_[rootB]) {
            parent_[rootB] = rootA;
        } else {
            parent_[rootB] = rootA;
            ++rank_[rootA];
        }
        return true;
    }

    /**
     * @brief Check whether @p a and @p b belong to the same set.
     * @throws std::out_of_range if a or b >= Size().
     */
    bool Connected(std::size_t a, std::size_t b) {
        return Find(a) == Find(b);
    }

    /** @brief Return the total number of elements. */
    [[nodiscard]] std::size_t Size() const noexcept { return parent_.size(); }

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> rank_;

    void ValidateIndex(std::size_t x) const {
        if (x >= parent_.size()) {
            throw std::out_of_range("DisjointSet: index out of range");
        }
    }
};
