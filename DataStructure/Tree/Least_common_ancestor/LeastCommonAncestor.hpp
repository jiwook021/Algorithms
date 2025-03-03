/**
 * @file LeastCommonAncestor.hpp
 * @brief Least Common Ancestor (LCA) using Euler tour + Sparse Table (RMQ).
 * @details Given a rooted tree, finds the lowest common ancestor of any two
 *          nodes in O(1) per query after O(n log n) preprocessing.
 *
 * Algorithm:
 *   1. Perform an Euler tour (DFS), recording each node visit and its depth.
 *   2. Record the first occurrence of each node in the Euler tour.
 *   3. An LCA query for (u, v) reduces to a Range Minimum Query on the
 *      depth array between the first occurrences of u and v.
 *   4. The Sparse Table answers RMQ in O(1) after O(n log n) build.
 *
 * Complexity:
 *   Build       : O(n log n) time and space
 *   Query (LCA) : O(1) per query
 *   AddEdge     : O(1)
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * @class LeastCommonAncestor
 * @brief Finds LCA of two nodes in a tree using Euler tour + Sparse Table.
 */
class LeastCommonAncestor {
public:
    /**
     * @brief Construct an LCA solver for a tree with @p nodeCount nodes (0-indexed).
     * @param nodeCount Number of nodes in the tree.
     * @throws std::invalid_argument if nodeCount is 0.
     */
    explicit LeastCommonAncestor(std::size_t nodeCount)
        : nodeCount_(nodeCount), adj_(nodeCount), built_(false) {
        if (nodeCount == 0) {
            throw std::invalid_argument("LeastCommonAncestor: nodeCount must be > 0");
        }
    }

    /**
     * @brief Add an undirected edge between nodes u and v.
     * @param u First node (0-indexed).
     * @param v Second node (0-indexed).
     */
    void AddEdge(std::size_t u, std::size_t v) {
        ValidateNode(u);
        ValidateNode(v);
        adj_[u].push_back(v);
        adj_[v].push_back(u);
        built_ = false;
    }

    /**
     * @brief Build the data structures for LCA queries.
     * @param root The root of the tree (0-indexed, default 0).
     *
     * Must be called after all edges are added and before any Query() call.
     */
    void Build(std::size_t root = 0) {
        ValidateNode(root);

        // Euler tour
        euler_.clear();
        depth_.clear();
        firstOccurrence_.assign(nodeCount_, 0);
        std::vector<bool> visited(nodeCount_, false);

        // Iterative DFS for Euler tour
        struct Frame {
            std::size_t node;
            std::size_t depthVal;
            std::size_t childIdx;
        };

        std::vector<Frame> stack;
        stack.push_back({root, 0, 0});
        visited[root] = true;
        euler_.push_back(root);
        depth_.push_back(0);
        firstOccurrence_[root] = 0;

        while (!stack.empty()) {
            auto& frame = stack.back();

            // Find next unvisited child
            bool foundChild = false;
            while (frame.childIdx < adj_[frame.node].size()) {
                std::size_t child = adj_[frame.node][frame.childIdx];
                ++frame.childIdx;
                if (!visited[child]) {
                    visited[child] = true;
                    firstOccurrence_[child] = euler_.size();
                    euler_.push_back(child);
                    depth_.push_back(frame.depthVal + 1);
                    stack.push_back({child, frame.depthVal + 1, 0});
                    foundChild = true;
                    break;
                }
            }

            if (!foundChild) {
                stack.pop_back();
                if (!stack.empty()) {
                    // Back-edge: re-record parent
                    euler_.push_back(stack.back().node);
                    depth_.push_back(stack.back().depthVal);
                }
            }
        }

        // Build sparse table on depth_ array
        BuildSparseTable();
        built_ = true;
    }

    /**
     * @brief Query the LCA of nodes u and v.
     * @param u First node (0-indexed).
     * @param v Second node (0-indexed).
     * @return The LCA node index.
     * @throws std::logic_error if Build() has not been called.
     */
    std::size_t Query(std::size_t u, std::size_t v) const {
        if (!built_) {
            throw std::logic_error("LeastCommonAncestor: Build() must be called first");
        }
        ValidateNode(u);
        ValidateNode(v);

        std::size_t left = firstOccurrence_[u];
        std::size_t right = firstOccurrence_[v];
        if (left > right) {
            std::size_t tmp = left;
            left = right;
            right = tmp;
        }

        return RangeMinQuery(left, right);
    }

    /** @brief Return the number of nodes. */
    [[nodiscard]] std::size_t NodeCount() const noexcept { return nodeCount_; }

private:
    std::size_t nodeCount_;
    std::vector<std::vector<std::size_t>> adj_;
    bool built_;

    // Euler tour data
    std::vector<std::size_t> euler_;    ///< Euler tour node sequence
    std::vector<std::size_t> depth_;    ///< Depth at each Euler tour position
    std::vector<std::size_t> firstOccurrence_;  ///< First occurrence of each node in euler_

    // Sparse table for RMQ
    std::vector<std::vector<std::size_t>> sparseTable_;  ///< sparseTable_[k][i] = index of min depth in [i, i+2^k)
    std::vector<int> log2Table_;

    void ValidateNode(std::size_t node) const {
        if (node >= nodeCount_) {
            throw std::out_of_range("LeastCommonAncestor: node index out of range");
        }
    }

    /** @brief Build the sparse table for O(1) RMQ on the depth array. */
    void BuildSparseTable() {
        std::size_t n = euler_.size();
        if (n == 0) return;

        // Precompute log2
        log2Table_.assign(n + 1, 0);
        for (std::size_t i = 2; i <= n; ++i) {
            log2Table_[i] = log2Table_[i / 2] + 1;
        }

        int maxLog = log2Table_[n] + 1;
        sparseTable_.assign(maxLog, std::vector<std::size_t>(n, 0));

        // Base case: intervals of length 1
        for (std::size_t i = 0; i < n; ++i) {
            sparseTable_[0][i] = i;
        }

        // Fill for larger intervals
        for (int k = 1; k < maxLog; ++k) {
            for (std::size_t i = 0; i + (1u << k) <= n; ++i) {
                std::size_t left = sparseTable_[k - 1][i];
                std::size_t right = sparseTable_[k - 1][i + (1u << (k - 1))];
                sparseTable_[k][i] = (depth_[left] <= depth_[right]) ? left : right;
            }
        }
    }

    /** @brief O(1) Range Minimum Query on the depth array. Returns the node at minimum depth. */
    std::size_t RangeMinQuery(std::size_t left, std::size_t right) const {
        int k = log2Table_[right - left + 1];
        std::size_t a = sparseTable_[k][left];
        std::size_t b = sparseTable_[k][right - (1u << k) + 1];
        return euler_[(depth_[a] <= depth_[b]) ? a : b];
    }
};
