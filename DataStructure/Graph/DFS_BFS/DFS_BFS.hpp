/**
 * @file DFS_BFS.hpp
 * @brief Graph traversal: depth-first and breadth-first search
 * @details Adjacency-list graph with recursive DFS and iterative BFS.
 *          Supports directed/undirected edges and tracks visited nodes
 *          for cycle-safe traversal. Returns traversal order as vectors
 *          for testability.
 */

#pragma once

#include <cstddef>
#include <queue>
#include <stdexcept>
#include <vector>

/**
 * @class Graph
 * @brief Undirected adjacency-list graph supporting DFS and BFS traversal.
 *
 * Complexity:
 *   DFS / BFS : O(V + E)
 *   AddEdge   : O(1)
 *   Space     : O(V + E)
 */
class Graph {
public:
    /**
     * @brief Construct a graph with @p nodes vertices (0-indexed).
     * @throws std::invalid_argument if nodes <= 0.
     */
    explicit Graph(int nodes) {
        if (nodes <= 0) {
            throw std::invalid_argument("Number of nodes must be positive.");
        }
        adjacencyList_.resize(static_cast<std::size_t>(nodes));
        nodeCount_ = nodes;
    }

    /**
     * @brief Add an undirected edge between u and v.
     * @throws std::out_of_range if u or v is out of bounds.
     */
    void AddEdge(int u, int v) {
        if (u >= nodeCount_ || v >= nodeCount_ || u < 0 || v < 0) {
            throw std::out_of_range("Invalid node index for edge.");
        }
        adjacencyList_[static_cast<std::size_t>(u)].push_back(v);
        adjacencyList_[static_cast<std::size_t>(v)].push_back(u);
    }

    /**
     * @brief Perform depth-first search starting from @p startNode.
     * @return Vector of visited node indices in DFS order.
     * @throws std::invalid_argument if startNode is out of bounds.
     */
    std::vector<int> DFS(int startNode) {
        if (startNode < 0 || startNode >= nodeCount_) {
            throw std::invalid_argument("Invalid start node for DFS.");
        }
        std::vector<bool> visited(static_cast<std::size_t>(nodeCount_), false);
        std::vector<int> order;
        DfsRecursive(startNode, visited, order);
        return order;
    }

    /**
     * @brief Perform breadth-first search starting from @p startNode.
     * @return Vector of visited node indices in BFS order.
     * @throws std::invalid_argument if startNode is out of bounds.
     */
    std::vector<int> BFS(int startNode) {
        if (startNode < 0 || startNode >= nodeCount_) {
            throw std::invalid_argument("Invalid start node for BFS.");
        }
        std::vector<bool> visited(static_cast<std::size_t>(nodeCount_), false);
        std::vector<int> order;
        std::queue<int> q;

        visited[static_cast<std::size_t>(startNode)] = true;
        q.push(startNode);

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            order.push_back(current);

            for (int neighbor : adjacencyList_[static_cast<std::size_t>(current)]) {
                if (!visited[static_cast<std::size_t>(neighbor)]) {
                    visited[static_cast<std::size_t>(neighbor)] = true;
                    q.push(neighbor);
                }
            }
        }
        return order;
    }

    /** @brief Return the number of vertices. */
    [[nodiscard]] int NodeCount() const noexcept { return nodeCount_; }

private:
    int nodeCount_;
    std::vector<std::vector<int>> adjacencyList_;

    void DfsRecursive(int node, std::vector<bool>& visited, std::vector<int>& order) {
        visited[static_cast<std::size_t>(node)] = true;
        order.push_back(node);
        for (int neighbor : adjacencyList_[static_cast<std::size_t>(node)]) {
            if (!visited[static_cast<std::size_t>(neighbor)]) {
                DfsRecursive(neighbor, visited, order);
            }
        }
    }
};
