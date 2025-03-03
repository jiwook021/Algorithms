/**
 * @file main.cpp
 * @brief Driver for Kruskal's MST algorithm.
 */

#include "KruskalAlgorithm.hpp"
#include <iostream>

int main() {
    // Build a 6-vertex weighted graph (vertices 0=A through 5=F):
    //
    //        9          12
    //   A -------- B         A ---------- C
    //   |          |  2          8
    //  11          | B---C
    //   |          | 7  / 6    A --- D --- C
    //   F     D----E   /             6
    //    \   4  3   C---D     F --- D --- E
    //     F --- E  (13)             4     3
    //
    //  Expected MST edges: B-C(2), D-E(3), F-D(4), D-C(6), A-D(8)
    //  Total MST weight: 23

    KruskalGraph graph(6);
    graph.AddEdge(0, 1, 9);   // A-B
    graph.AddEdge(1, 2, 2);   // B-C
    graph.AddEdge(0, 2, 12);  // A-C
    graph.AddEdge(0, 3, 8);   // A-D
    graph.AddEdge(3, 2, 6);   // D-C
    graph.AddEdge(0, 5, 11);  // A-F
    graph.AddEdge(5, 3, 4);   // F-D
    graph.AddEdge(3, 4, 3);   // D-E
    graph.AddEdge(4, 2, 7);   // E-C
    graph.AddEdge(5, 4, 13);  // F-E

    auto mst = graph.ComputeMST();

    std::cout << "Kruskal MST edges:\n";
    int totalWeight = 0;
    for (const Edge& e : mst) {
        char src = static_cast<char>('A' + e.src);
        char dst = static_cast<char>('A' + e.dst);
        std::cout << "  " << src << "-" << dst << " (weight: " << e.weight << ")\n";
        totalWeight += e.weight;
    }
    std::cout << "Total MST weight: " << totalWeight << "\n";

    return 0;
}
