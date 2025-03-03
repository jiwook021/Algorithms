/**
 * @file Dijkstra.cpp
 * @brief Implementation of Graph with Dijkstra and Bellman-Ford.
 */

#include "Dijkstra.hpp"

#include <functional>
#include <queue>
#include <string>
#include <utility>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <typename Weight>
Graph<Weight>::Graph(std::size_t num_vertices)
    : num_vertices_{num_vertices},
      adjacency_list_(num_vertices + 1) {}

// ---------------------------------------------------------------------------
// Edge manipulation
// ---------------------------------------------------------------------------

template <typename Weight>
void Graph<Weight>::AddEdge(std::size_t from, std::size_t to, Weight weight,
                            bool undirected) {
    adjacency_list_[from].push_back({to, weight});
    if (undirected) {
        adjacency_list_[to].push_back({from, weight});
    }
    edges_.push_back({from, to, weight});
    if (undirected) {
        edges_.push_back({to, from, weight});
    }
}

// ---------------------------------------------------------------------------
// Dijkstra — shortest path with non-negative weights
// ---------------------------------------------------------------------------

template <typename Weight>
std::vector<Weight> Graph<Weight>::Dijkstra(std::size_t source) const {
    std::vector<Weight> dist(num_vertices_ + 1, kInfinity);
    dist[source] = Weight{0};

    // Min-heap ordered by (distance, vertex).
    // The smallest distance is processed first.
    std::priority_queue<std::pair<Weight, std::size_t>,
                        std::vector<std::pair<Weight, std::size_t>>,
                        std::greater<>> pq;
    pq.push({Weight{0}, source});

    while (!pq.empty()) {
        auto [cost, u] = pq.top();
        pq.pop();

        // Skip stale entries (we already found a shorter path).
        if (cost > dist[u]) continue;

        // Try to shorten the path to each neighbor of u.
        for (const auto& edge : adjacency_list_[u]) {
            Weight new_dist = cost + edge.weight;
            if (new_dist < dist[edge.to]) {
                dist[edge.to] = new_dist;
                pq.push({new_dist, edge.to});
            }
        }
    }
    return dist;
}

// ---------------------------------------------------------------------------
// Bellman-Ford — shortest path with negative weights allowed
// ---------------------------------------------------------------------------

template <typename Weight>
std::vector<Weight> Graph<Weight>::BellmanFord(std::size_t source) const {
    std::vector<Weight> dist(num_vertices_ + 1, kInfinity);
    dist[source] = Weight{0};

    // Relax all edges (V-1) times.  After round k, the shortest
    // path using at most k edges is correct.
    for (std::size_t round = 1; round < num_vertices_; ++round) {
        for (const auto& e : edges_) {
            if (dist[e.src] != kInfinity &&
                dist[e.src] + e.weight < dist[e.dst]) {
                dist[e.dst] = dist[e.src] + e.weight;
            }
        }
    }

    // One more pass: if any distance still decreases, there is
    // a negative-weight cycle reachable from the source.
    for (const auto& e : edges_) {
        if (dist[e.src] != kInfinity &&
            dist[e.src] + e.weight < dist[e.dst]) {
            return {};  // negative cycle detected
        }
    }
    return dist;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

template <typename Weight>
std::size_t Graph<Weight>::NumVertices() const {
    return num_vertices_;
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template class Graph<int>;
template class Graph<double>;
