/**
 * @file main.cpp
 * @brief Driver for LeastCommonAncestor (Euler tour + Sparse Table RMQ)
 */

#include "LeastCommonAncestor.hpp"
#include <iostream>

int main() {
    // Build a tree:
    //        0
    //       / |
    //      1   2
    //     / |   |
    //    3   4   5
    //   /
    //  6
    LeastCommonAncestor lca(7);
    lca.AddEdge(0, 1);
    lca.AddEdge(0, 2);
    lca.AddEdge(1, 3);
    lca.AddEdge(1, 4);
    lca.AddEdge(2, 5);
    lca.AddEdge(3, 6);

    lca.Build(0);

    std::cout << "Tree structure:\n";
    std::cout << "        0\n";
    std::cout << "       / \\\n";
    std::cout << "      1   2\n";
    std::cout << "     / \\   \\\n";
    std::cout << "    3   4   5\n";
    std::cout << "   /\n";
    std::cout << "  6\n\n";

    std::cout << "LCA(3, 4) = " << lca.Query(3, 4) << "\n";  // 1
    std::cout << "LCA(6, 4) = " << lca.Query(6, 4) << "\n";  // 1
    std::cout << "LCA(6, 5) = " << lca.Query(6, 5) << "\n";  // 0
    std::cout << "LCA(3, 5) = " << lca.Query(3, 5) << "\n";  // 0
    std::cout << "LCA(3, 3) = " << lca.Query(3, 3) << "\n";  // 3

    return 0;
}
