/**
 * @file DfsBfsCpp.hpp
 * @brief Functional-style DFS/BFS graph traversal using adjacency list vectors
 * @details Free-function based approach to DFS and BFS traversal, demonstrating
 *          a lightweight alternative to the class-based Graph in DFS_BFS/.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <queue>
#include <vector>

/**
 * @brief Add an undirected edge between source and dest in the adjacency list.
 * @param source Source vertex.
 * @param dest Destination vertex.
 * @param graph Adjacency list representation.
 */
inline void AddGraph(int source, int dest, std::vector<std::vector<int>>& graph) {
    graph[static_cast<std::size_t>(source)].push_back(dest);
    graph[static_cast<std::size_t>(dest)].push_back(source);
}

/**
 * @brief Perform DFS traversal and return visited node order.
 * @param index Starting vertex.
 * @param graph Adjacency list.
 * @return Vector of node indices in DFS visit order.
 */
inline std::vector<int> DfsCollect(int index, std::vector<std::vector<int>>& graph) {
    std::vector<int> result;
    std::vector<bool> found(graph.size(), false);
    std::function<void(int)> dfs = [&](int i) {
        found[static_cast<std::size_t>(i)] = true;
        result.push_back(i);
        for (int n : graph[static_cast<std::size_t>(i)]) {
            if (!found[static_cast<std::size_t>(n)]) dfs(n);
        }
    };
    dfs(index);
    return result;
}

/**
 * @brief Perform BFS traversal and return visited node order.
 * @param index Starting vertex.
 * @param graph Adjacency list.
 * @return Vector of node indices in BFS visit order.
 */
inline std::vector<int> BfsCollect(int index, std::vector<std::vector<int>>& graph) {
    std::vector<int> result;
    std::vector<bool> found(graph.size(), false);
    std::queue<int> q;
    q.push(index);
    found[static_cast<std::size_t>(index)] = true;
    while (!q.empty()) {
        int n = q.front();
        q.pop();
        result.push_back(n);
        for (int adj : graph[static_cast<std::size_t>(n)]) {
            if (!found[static_cast<std::size_t>(adj)]) {
                found[static_cast<std::size_t>(adj)] = true;
                q.push(adj);
            }
        }
    }
    return result;
}
