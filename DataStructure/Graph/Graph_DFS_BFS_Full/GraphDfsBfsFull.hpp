/**
 * @file GraphDfsBfsFull.hpp
 * @brief Full-featured adjacency-list graph with iterative DFS and BFS traversal.
 *
 * @details
 * Implements an undirected graph using sorted adjacency lists. Provides
 * both iterative DFS (using an explicit stack) and BFS (using a queue),
 * returning the traversal order as a vector for testability.
 *
 * Neighbor lists are kept in sorted order so that DFS and BFS always
 * visit the smallest-numbered neighbor first, making traversal results
 * deterministic and easy to verify.
 *
 * Complexity:
 *   DFS / BFS : O(V + E)
 *   AddEdge   : O(degree) — inserts into sorted neighbor list
 *   Space     : O(V + E)
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <queue>
#include <span>
#include <stack>
#include <stdexcept>
#include <vector>

/**
 * @class GraphFull
 * @brief Undirected adjacency-list graph with sorted neighbors and iterative traversal.
 *
 * Vertex indices are 0-based. Neighbor lists are kept sorted so that
 * traversal order is deterministic (smallest neighbor first).
 */
class GraphFull {
public:
    /**
     * @brief Construct a graph with @p num_vertices vertices (0-indexed).
     * @param num_vertices Number of vertices.
     * @throws std::invalid_argument if num_vertices is 0.
     */
    explicit GraphFull(std::size_t num_vertices);

    /**
     * @brief Add an undirected edge between u and v (sorted insertion).
     * @param u First vertex.
     * @param v Second vertex.
     * @throws std::out_of_range if u or v is out of bounds.
     */
    void AddEdge(std::size_t u, std::size_t v);

    /**
     * @brief Iterative depth-first search starting from @p start.
     *
     * Uses an explicit stack. At each vertex, neighbors are explored in
     * ascending order (smallest index first) due to sorted adjacency lists.
     *
     * @param start The starting vertex.
     * @return Vector of visited vertex indices in DFS order.
     * @throws std::out_of_range if start is out of bounds.
     */
    [[nodiscard]] std::vector<std::size_t> DFS(std::size_t start) const;

    /**
     * @brief Breadth-first search starting from @p start.
     *
     * Uses a queue. Neighbors are enqueued in ascending order due to
     * sorted adjacency lists.
     *
     * @param start The starting vertex.
     * @return Vector of visited vertex indices in BFS order.
     * @throws std::out_of_range if start is out of bounds.
     */
    [[nodiscard]] std::vector<std::size_t> BFS(std::size_t start) const;

    /**
     * @brief Return a span view of the sorted neighbor list for vertex @p vertex.
     *
     * std::span provides a zero-copy, non-owning view into the internal
     * adjacency vector -- callers can iterate without copying.
     *
     * @param vertex The vertex to query.
     * @return Read-only span over the neighbor indices.
     * @throws std::out_of_range if vertex is out of bounds.
     */
    [[nodiscard]] std::span<const std::size_t> Neighbors(std::size_t vertex) const;

    /** @brief Return the number of vertices. */
    [[nodiscard]] std::size_t NumVertices() const noexcept;

    /** @brief Return the number of edges. */
    [[nodiscard]] std::size_t NumEdges() const noexcept;

private:
    void ValidateVertex(std::size_t v) const;

    std::size_t num_vertices_;
    std::size_t num_edges_{0};
    std::vector<std::vector<std::size_t>> adjacency_list_;
};
