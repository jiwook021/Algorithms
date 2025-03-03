/**
 * @file main.cpp
 * @brief Interactive driver for graph DFS and BFS traversal.
 *
 * Builds a sample graph edge by edge, shows the adjacency list,
 * runs DFS and BFS with step-by-step output, then lets the user
 * add edges and run traversals interactively.
 */

#include "GraphDfsBfsFull.hpp"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Print the full adjacency list of the graph.
 */
static void PrintGraph(const GraphFull& graph) {
    std::cout << "  Adjacency List:\n";
    for (std::size_t v = 0; v < graph.NumVertices(); ++v) {
        std::cout << "    " << v << " -> [";
        bool first = true;
        for (std::size_t n : graph.Neighbors(v)) {
            if (!first) std::cout << ", ";
            std::cout << n;
            first = false;
        }
        std::cout << "]\n";
    }
}

/**
 * @brief Print a traversal result with a label.
 */
static void PrintTraversal(const std::string& name,
                           const std::vector<std::size_t>& order) {
    std::cout << "  " << name << ": ";
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (i > 0) std::cout << " -> ";
        std::cout << order[i];
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=============================================\n"
              << "  Graph Traversal: DFS and BFS\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  DFS and BFS are two ways to visit every vertex in a\n"
              << "  graph starting from a given vertex.\n\n"

              << "DFS (Depth-First Search)\n"
              << "  Goes as deep as possible along each path before\n"
              << "  backtracking.  Uses a stack (last-in, first-out).\n"
              << "  Think: exploring a maze by following one path to\n"
              << "  a dead end, then going back to try the next turn.\n\n"

              << "BFS (Breadth-First Search)\n"
              << "  Visits all neighbors first, then neighbors-of-\n"
              << "  neighbors, and so on.  Uses a queue (first-in,\n"
              << "  first-out).  Think: a ripple spreading outward\n"
              << "  from a stone dropped in water.\n\n"

              << "KEY DIFFERENCE\n"
              << "  DFS explores far away vertices first (depth).\n"
              << "  BFS explores nearby vertices first (breadth).\n"
              << "  BFS always finds the shortest path (fewest edges).\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Build a 6-vertex graph, showing each edge as it is added
    //
    //     0 --- 1 --- 2
    //     |     |     |
    //     3 --- 4 --- 5
    //
    // -----------------------------------------------------------------

    std::cout << "BUILDING A GRAPH (6 vertices)\n\n"
              << "  Target shape:\n"
              << "      0 --- 1 --- 2\n"
              << "      |     |     |\n"
              << "      3 --- 4 --- 5\n\n";

    GraphFull graph(6);

    // Define edges and add them one by one with output
    struct Edge { std::size_t u, v; };
    std::vector<Edge> edges = {
        {0, 1}, {1, 2}, {0, 3}, {1, 4}, {2, 5}, {3, 4}, {4, 5},
    };

    std::cout << "  Adding edges:\n";
    for (const auto& e : edges) {
        graph.AddEdge(e.u, e.v);
        std::cout << "    AddEdge(" << e.u << ", " << e.v
                  << ")  ->  " << e.u << " -- " << e.v
                  << "  (total edges: " << graph.NumEdges() << ")\n";
    }

    std::cout << "\n  Graph: " << graph.NumVertices() << " vertices, "
              << graph.NumEdges() << " edges\n\n";

    PrintGraph(graph);

    // -----------------------------------------------------------------
    // Run DFS from vertex 0
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  DFS from vertex 0\n"
              << "---------------------------------------------\n\n"
              << "  Strategy: go deep first, backtrack when stuck.\n"
              << "  At each vertex, visit the smallest unvisited neighbor.\n\n";

    auto dfs_order = graph.DFS(0);
    PrintTraversal("DFS order", dfs_order);

    std::cout << "\n  DFS explores a path as far as it can go:\n"
              << "    ";
    for (std::size_t i = 0; i < dfs_order.size(); ++i) {
        if (i > 0) std::cout << " -> ";
        std::cout << dfs_order[i];
    }
    std::cout << "\n";

    // -----------------------------------------------------------------
    // Run BFS from vertex 0
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  BFS from vertex 0\n"
              << "---------------------------------------------\n\n"
              << "  Strategy: visit all neighbors first, then go one\n"
              << "  level deeper.  Finds shortest paths.\n\n";

    auto bfs_order = graph.BFS(0);
    PrintTraversal("BFS order", bfs_order);

    // Show BFS levels
    std::cout << "\n  BFS visits vertices level by level:\n"
              << "    Level 0 (start)     : " << bfs_order[0] << "\n"
              << "    Level 1 (neighbors) : ";
    // Neighbors of 0 are 1 and 3
    bool first = true;
    for (std::size_t i = 1; i < bfs_order.size(); ++i) {
        // A simple heuristic: vertices at distance 1 from start
        bool is_neighbor = false;
        for (std::size_t n : graph.Neighbors(bfs_order[0])) {
            if (bfs_order[i] == n) { is_neighbor = true; break; }
        }
        if (is_neighbor) {
            if (!first) std::cout << ", ";
            std::cout << bfs_order[i];
            first = false;
        }
    }
    std::cout << "\n    Level 2 (2 hops)    : ";
    first = true;
    for (std::size_t i = 1; i < bfs_order.size(); ++i) {
        bool is_neighbor = false;
        for (std::size_t n : graph.Neighbors(bfs_order[0])) {
            if (bfs_order[i] == n) { is_neighbor = true; break; }
        }
        if (!is_neighbor) {
            if (!first) std::cout << ", ";
            std::cout << bfs_order[i];
            first = false;
        }
    }
    std::cout << "\n";

    // -----------------------------------------------------------------
    // Compare DFS vs BFS
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Comparison\n"
              << "---------------------------------------------\n\n";
    PrintTraversal("DFS", dfs_order);
    PrintTraversal("BFS", bfs_order);
    std::cout << "\n  DFS went deep (followed one path far).\n"
              << "  BFS spread out (visited nearby vertices first).\n";

    // -----------------------------------------------------------------
    // Interactive mode
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Interactive Mode\n"
              << "  Commands:\n"
              << "    add U V    — add an edge between U and V\n"
              << "    dfs START  — run DFS from vertex START\n"
              << "    bfs START  — run BFS from vertex START\n"
              << "    show       — print the adjacency list\n"
              << "    q          — quit\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  > ";
        std::string cmd;
        std::getline(std::cin, cmd);

        if (cmd.empty() || cmd[0] == 'q' || cmd[0] == 'Q') {
            break;
        }

        if (cmd.rfind("add", 0) == 0) {
            try {
                std::size_t pos = 3;
                unsigned long u = std::stoul(cmd.substr(pos), &pos);
                unsigned long v = std::stoul(cmd.substr(3 + pos));
                graph.AddEdge(u, v);
                std::cout << "    Added edge " << u << " -- " << v
                          << "  (total edges: " << graph.NumEdges() << ")\n";
                PrintGraph(graph);
            } catch (...) {
                std::cout << "    Invalid. Usage: add U V  (e.g. add 0 5)\n";
            }
        } else if (cmd.rfind("dfs", 0) == 0) {
            try {
                unsigned long start = std::stoul(cmd.substr(3));
                auto order = graph.DFS(start);
                PrintTraversal("DFS", order);
            } catch (...) {
                std::cout << "    Invalid. Usage: dfs START  (e.g. dfs 0)\n";
            }
        } else if (cmd.rfind("bfs", 0) == 0) {
            try {
                unsigned long start = std::stoul(cmd.substr(3));
                auto order = graph.BFS(start);
                PrintTraversal("BFS", order);
            } catch (...) {
                std::cout << "    Invalid. Usage: bfs START  (e.g. bfs 0)\n";
            }
        } else if (cmd.rfind("show", 0) == 0) {
            PrintGraph(graph);
        } else {
            std::cout << "    Unknown command. Try: add, dfs, bfs, show, q\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
