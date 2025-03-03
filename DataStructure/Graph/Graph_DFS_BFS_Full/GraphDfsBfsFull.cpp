/**
 * @file GraphDfsBfsFull.cpp
 * @brief Implementation of GraphFull: undirected graph with iterative DFS/BFS.
 */

#include "GraphDfsBfsFull.hpp"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

GraphFull::GraphFull(std::size_t num_vertices)
    : num_vertices_(num_vertices), adjacency_list_(num_vertices) {
    if (num_vertices == 0) {
        throw std::invalid_argument("GraphFull: number of vertices must be positive.");
    }
}

// ---------------------------------------------------------------------------
// Edge manipulation
// ---------------------------------------------------------------------------

void GraphFull::AddEdge(std::size_t u, std::size_t v) {
    ValidateVertex(u);
    ValidateVertex(v);

    // Insert the neighbor into the adjacency list at the correct position
    // so the list stays sorted.  This way DFS/BFS always visit the
    // smallest-numbered neighbor first.
    //
    // Example: if vertex 0 already has neighbors [1, 3] and we add
    // edge 0-2, lower_bound finds the slot between 1 and 3, producing [1, 2, 3].
    auto insert_sorted = [](std::vector<std::size_t>& neighbors,
                            std::size_t new_neighbor) {
        auto pos = std::lower_bound(neighbors.begin(), neighbors.end(),
                                    new_neighbor);
        neighbors.insert(pos, new_neighbor);
    };

    insert_sorted(adjacency_list_[u], v);   // u's neighbor list gets v
    insert_sorted(adjacency_list_[v], u);   // v's neighbor list gets u
    ++num_edges_;
}

// ---------------------------------------------------------------------------
// Traversal -- DFS (iterative)
// ---------------------------------------------------------------------------

std::vector<std::size_t> GraphFull::DFS(std::size_t start) const {
    ValidateVertex(start);

    std::vector<bool> visited(num_vertices_, false);
    std::vector<std::size_t> order;
    std::stack<std::size_t> dfs_stack;

    visited[start] = true;
    order.push_back(start);
    dfs_stack.push(start);

    while (!dfs_stack.empty()) {
        std::size_t current = dfs_stack.top();
        bool found = false;

        for (std::size_t neighbor : adjacency_list_[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                order.push_back(neighbor);
                dfs_stack.push(neighbor);
                found = true;
                break;  // Go deeper immediately
            }
        }

        if (!found) {
            dfs_stack.pop();  // Backtrack
        }
    }

    return order;
}

// ---------------------------------------------------------------------------
// Traversal -- BFS
// ---------------------------------------------------------------------------

std::vector<std::size_t> GraphFull::BFS(std::size_t start) const {
    ValidateVertex(start);

    std::vector<bool> visited(num_vertices_, false);
    std::vector<std::size_t> order;
    std::queue<std::size_t> bfs_queue;

    visited[start] = true;
    bfs_queue.push(start);

    while (!bfs_queue.empty()) {
        std::size_t current = bfs_queue.front();
        bfs_queue.pop();
        order.push_back(current);

        for (std::size_t neighbor : adjacency_list_[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                bfs_queue.push(neighbor);
            }
        }
    }

    return order;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

std::span<const std::size_t> GraphFull::Neighbors(std::size_t vertex) const {
    ValidateVertex(vertex);
    return std::span<const std::size_t>(adjacency_list_[vertex]);
}

std::size_t GraphFull::NumVertices() const noexcept {
    return num_vertices_;
}

std::size_t GraphFull::NumEdges() const noexcept {
    return num_edges_;
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

void GraphFull::ValidateVertex(std::size_t v) const {
    if (v >= num_vertices_) {
        throw std::out_of_range("GraphFull: vertex index out of range.");
    }
}
