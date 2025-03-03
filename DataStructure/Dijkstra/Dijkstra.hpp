/**
 * @file Dijkstra.hpp
 * @brief Weighted graph with Dijkstra and Bellman-Ford shortest path.
 *
 * @details
 * Given a graph where each edge has a weight (distance, cost, time),
 * shortest-path algorithms find the path with the smallest total
 * weight from a starting vertex to every other vertex.
 *
 * Two algorithms are provided:
 *   Dijkstra     — fast (O(V log V + E) with a min-heap), but only
 *                  works when all edge weights are non-negative.
 *   Bellman-Ford — slower (O(V * E)), but handles negative weights
 *                  and can detect negative-weight cycles.
 *
 * Vertices are 1-indexed: a graph with N vertices uses indices 1..N.
 * Unreachable vertices have distance = kInfinity (max value of the
 * weight type).
 */

#pragma once

#include <cstddef>
#include <limits>
#include <vector>

/**
 * @class Graph
 * @brief Weighted graph with Dijkstra and Bellman-Ford shortest path.
 *
 * @tparam Weight Numeric type for edge weights (int, double, etc.).
 */
template <typename Weight = int>
class Graph {
public:
    /// Sentinel representing "no path exists" (max value of Weight).
    static constexpr Weight kInfinity = std::numeric_limits<Weight>::max();

    /**
     * @brief Construct a graph with @p num_vertices vertices (1-indexed).
     */
    explicit Graph(std::size_t num_vertices);

    /**
     * @brief Add an edge between @p from and @p to with the given weight.
     *
     * If @p undirected is true (default), both directions are added.
     */
    void AddEdge(std::size_t from, std::size_t to, Weight weight,
                 bool undirected = true);

    /**
     * @brief Find shortest distances from @p source to all vertices.
     *
     * Uses a priority queue (min-heap) to always process the closest
     * unvisited vertex next.  Only works with non-negative weights.
     *
     * @return Vector of distances indexed by vertex number.
     */
    [[nodiscard]] std::vector<Weight> Dijkstra(std::size_t source) const;

    /**
     * @brief Find shortest distances from @p source to all vertices.
     *
     * Handles negative weights.  Detects negative-weight cycles.
     *
     * @return Vector of distances, or empty vector if a negative
     *         cycle is reachable from source.
     */
    [[nodiscard]] std::vector<Weight> BellmanFord(std::size_t source) const;

    /** @brief Return the number of vertices. */
    [[nodiscard]] std::size_t NumVertices() const;

private:
    struct Edge {
        std::size_t to;
        Weight weight;
    };

    struct BfEdge {
        std::size_t src;
        std::size_t dst;
        Weight weight;
    };

    std::size_t num_vertices_;
    std::vector<std::vector<Edge>> adjacency_list_;
    std::vector<BfEdge> edges_;
};
