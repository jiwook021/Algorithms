/**
 * @file main.cpp
 * @brief Driver for DFS_BFS graph traversal
 */

#include "DFS_BFS.hpp"
#include <iostream>

int main() {
    Graph g(6);
    g.AddEdge(0, 1);
    g.AddEdge(0, 2);
    g.AddEdge(1, 3);
    g.AddEdge(1, 4);
    g.AddEdge(2, 5);

    auto dfsOrder = g.DFS(0);
    std::cout << "DFS traversal: ";
    for (int v : dfsOrder) std::cout << v << " ";
    std::cout << "\n";

    auto bfsOrder = g.BFS(0);
    std::cout << "BFS traversal: ";
    for (int v : bfsOrder) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
