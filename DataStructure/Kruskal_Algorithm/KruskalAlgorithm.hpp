/**
 * @file KruskalAlgorithm.hpp
 * @brief Kruskal's Minimum Spanning Tree algorithm with Union-Find.
 *
 * @details
 * Implements Kruskal's algorithm for finding the Minimum Spanning Tree (MST)
 * of a weighted, undirected graph. Uses a Union-Find (Disjoint Set) data
 * structure with path compression and union by rank for efficient cycle
 * detection.
 *
 * Algorithm:
 *   1. Sort all edges by weight (ascending).
 *   2. Initialize each vertex in its own disjoint set.
 *   3. For each edge (u, v, w) in sorted order:
 *      - If u and v are in different sets, add the edge to the MST and
 *        merge the sets.
 *      - If they are already connected, skip (would create a cycle).
 *   4. Stop after selecting V-1 edges (a tree on V vertices).
 *
 * Complexity:
 *   Time:  O(E log E) -- dominated by sorting edges
 *   Space: O(V + E)
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

/**
 * @struct Edge
 * @brief Weighted edge in an undirected graph.
 */
struct Edge {
    int src;     ///< Source vertex
    int dst;     ///< Destination vertex
    int weight;  ///< Edge weight

    /** @brief Compare edges by weight (for sorting). */
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

/**
 * @class DisjointSetKruskal
 * @brief Union-Find with path compression and union by rank.
 *
 * Used internally by Kruskal's algorithm for O(alpha(n)) cycle detection.
 */
class DisjointSetKruskal {
public:
    /**
     * @brief Construct a disjoint set with @p n elements, each in its own set.
     * @param n Number of elements (indexed 0..n-1).
     */
    explicit DisjointSetKruskal(std::size_t n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), static_cast<std::size_t>(0));
    }

    /**
     * @brief Find the representative of the set containing @p x.
     * @param x Element index.
     * @return Root of x's set (with path compression applied).
     */
    std::size_t Find(std::size_t x) {
        if (parent_[x] != x) {
            parent_[x] = Find(parent_[x]);
        }
        return parent_[x];
    }

    /**
     * @brief Merge the sets containing @p a and @p b.
     * @return true if a merge occurred, false if already in the same set.
     */
    bool Union(std::size_t a, std::size_t b) {
        std::size_t rootA = Find(a);
        std::size_t rootB = Find(b);
        if (rootA == rootB) return false;

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

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> rank_;
};

/**
 * @class KruskalGraph
 * @brief Weighted undirected graph with Kruskal's MST algorithm.
 */
class KruskalGraph {
public:
    /**
     * @brief Construct a graph with @p numVertices vertices (0-indexed).
     * @param numVertices Number of vertices.
     * @throws std::invalid_argument if numVertices is 0.
     */
    explicit KruskalGraph(int numVertices)
        : numVertices_(numVertices) {
        if (numVertices <= 0) {
            throw std::invalid_argument("KruskalGraph: number of vertices must be positive.");
        }
    }

    /**
     * @brief Add an undirected weighted edge between u and v.
     * @param u First vertex.
     * @param v Second vertex.
     * @param weight Edge weight.
     * @throws std::out_of_range if u or v is out of bounds.
     */
    void AddEdge(int u, int v, int weight) {
        ValidateVertex(u);
        ValidateVertex(v);
        edges_.push_back({u, v, weight});
    }

    /**
     * @brief Compute the Minimum Spanning Tree using Kruskal's algorithm.
     * @return Vector of edges in the MST.
     *
     * The returned edges are sorted by weight. If the graph is disconnected,
     * the result contains a minimum spanning forest (fewer than V-1 edges).
     */
    std::vector<Edge> ComputeMST() const {
        std::vector<Edge> sortedEdges = edges_;
        std::sort(sortedEdges.begin(), sortedEdges.end());

        DisjointSetKruskal ds(static_cast<std::size_t>(numVertices_));
        std::vector<Edge> mst;
        mst.reserve(static_cast<std::size_t>(numVertices_ - 1));

        for (const Edge& e : sortedEdges) {
            if (ds.Union(static_cast<std::size_t>(e.src),
                         static_cast<std::size_t>(e.dst))) {
                mst.push_back(e);
                if (static_cast<int>(mst.size()) == numVertices_ - 1) {
                    break;  // MST complete
                }
            }
        }

        return mst;
    }

    /**
     * @brief Compute the total weight of the MST.
     * @return Sum of weights of MST edges.
     */
    int MSTWeight() const {
        auto mst = ComputeMST();
        int total = 0;
        for (const Edge& e : mst) {
            total += e.weight;
        }
        return total;
    }

    /** @brief Return the number of vertices. */
    [[nodiscard]] int NumVertices() const noexcept { return numVertices_; }

    /** @brief Return the number of edges. */
    [[nodiscard]] int NumEdges() const noexcept {
        return static_cast<int>(edges_.size());
    }

    /** @brief Return all edges. */
    const std::vector<Edge>& Edges() const noexcept { return edges_; }

private:
    int numVertices_;
    std::vector<Edge> edges_;

    void ValidateVertex(int v) const {
        if (v < 0 || v >= numVertices_) {
            throw std::out_of_range("KruskalGraph: vertex index out of range.");
        }
    }
};
