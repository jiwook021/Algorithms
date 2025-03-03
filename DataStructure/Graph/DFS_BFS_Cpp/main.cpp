/**
 * @file main.cpp
 * @brief Driver for DfsBfsCpp functional-style graph traversal
 */

#include "DfsBfsCpp.hpp"
#include <iostream>

int main() {
    std::vector<std::vector<int>> graph(5);

    AddGraph(0, 2, graph);
    AddGraph(0, 1, graph);
    AddGraph(0, 3, graph);
    AddGraph(2, 3, graph);
    AddGraph(3, 4, graph);

    auto dfsOrder = DfsCollect(0, graph);
    std::cout << "DFS: ";
    for (int v : dfsOrder) std::cout << v << " ";
    std::cout << "\n";

    auto bfsOrder = BfsCollect(0, graph);
    std::cout << "BFS: ";
    for (int v : bfsOrder) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
