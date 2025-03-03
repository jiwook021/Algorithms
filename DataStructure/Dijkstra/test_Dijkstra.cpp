/**
 * @file test_Dijkstra.cpp
 * @brief Google Test suite for the templatized Graph with Dijkstra and
 *        Bellman-Ford shortest-path algorithms.
 *
 * Each test prints an educational banner explaining *why* it matters.
 */

#include "Dijkstra.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <string>

namespace {

/**
 * @brief Print a section banner for educational output.
 * @param title Test section title.
 * @param why   One-line explanation of why this test matters.
 */
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

} // namespace

// ---------------------------------------------------------------------------
// Dijkstra -- basic shortest path
// ---------------------------------------------------------------------------

TEST(DijkstraTest, ShortestPathFromSource) {
    PrintSection("Shortest Path From Source",
                 "Dijkstra must compute the minimum-cost path to every "
                 "reachable vertex in a non-negative-weight graph.");

    //       2       4
    //   1 ───── 2 ───── 4
    //   │       │       │
    //   5       1       3
    //   │       │       │
    //   3 ───── 3       5
    //       1
    Graph<int> g(5);
    g.AddEdge(1, 2, 2);
    g.AddEdge(1, 3, 5);
    g.AddEdge(2, 3, 1);
    g.AddEdge(2, 4, 4);
    g.AddEdge(3, 4, 1);
    g.AddEdge(4, 5, 3);

    auto dist = g.Dijkstra(1);

    EXPECT_EQ(dist[1], 0);
    EXPECT_EQ(dist[2], 2);
    EXPECT_EQ(dist[3], 3);  // 1->2->3 = 3, shorter than 1->3 = 5
    EXPECT_EQ(dist[4], 4);  // 1->2->3->4 = 4
    EXPECT_EQ(dist[5], 7);  // 1->2->3->4->5 = 7

    std::cout << "dist[] = {";
    for (std::size_t i = 1; i <= 5; ++i) {
        std::cout << " " << dist[i];
    }
    std::cout << " }" << std::endl;
}

// ---------------------------------------------------------------------------
// Unreachable vertex returns infinity
// ---------------------------------------------------------------------------

TEST(DijkstraTest, UnreachableVertexReturnsInfinity) {
    PrintSection("Unreachable Vertex Returns Infinity",
                 "Vertices with no path from the source must retain the "
                 "sentinel value std::numeric_limits<Weight>::max().");

    Graph<int> g(3);
    g.AddEdge(1, 2, 5);
    // Vertex 3 has no edges at all.

    auto dist = g.Dijkstra(1);

    EXPECT_EQ(dist[1], 0);
    EXPECT_EQ(dist[2], 5);
    EXPECT_EQ(dist[3], Graph<int>::kInfinity);
    EXPECT_EQ(dist[3], std::numeric_limits<int>::max());

    std::cout << "dist[3] = " << dist[3]
              << " (numeric_limits<int>::max = "
              << std::numeric_limits<int>::max() << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// Bellman-Ford matches Dijkstra on non-negative weights
// ---------------------------------------------------------------------------

TEST(BellmanFordTest, MatchesDijkstraOnNonNegativeWeights) {
    PrintSection("Bellman-Ford Matches Dijkstra",
                 "On graphs with only non-negative weights, both algorithms "
                 "must produce identical shortest-path distances.");

    Graph<int> g(5);
    g.AddEdge(1, 2, 2);
    g.AddEdge(1, 3, 5);
    g.AddEdge(2, 3, 1);
    g.AddEdge(2, 4, 4);
    g.AddEdge(3, 4, 1);
    g.AddEdge(4, 5, 3);

    auto dijkstra_dist = g.Dijkstra(1);
    auto bf_dist       = g.BellmanFord(1);

    ASSERT_FALSE(bf_dist.empty()) << "No negative cycle expected.";

    for (std::size_t i = 1; i <= g.NumVertices(); ++i) {
        EXPECT_EQ(dijkstra_dist[i], bf_dist[i])
            << "Mismatch at vertex " << i;
    }

    std::cout << "All " << g.NumVertices()
              << " vertices match between Dijkstra and Bellman-Ford."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Bellman-Ford handles negative weights
// ---------------------------------------------------------------------------

TEST(BellmanFordTest, HandlesNegativeWeights) {
    PrintSection("Bellman-Ford Handles Negative Weights",
                 "Unlike Dijkstra, Bellman-Ford correctly relaxes edges with "
                 "negative weights as long as no negative cycle exists.");

    // Directed graph with a negative edge:
    //   1 --10--> 2 ---(-5)---> 3
    //   |                       ^
    //   +--------7--------------+
    Graph<int> g(3);
    g.AddEdge(1, 2, 10, false);  // directed
    g.AddEdge(2, 3, -5, false);  // negative weight
    g.AddEdge(1, 3, 7,  false);  // direct but longer path

    auto dist = g.BellmanFord(1);

    ASSERT_FALSE(dist.empty()) << "No negative cycle in this graph.";
    EXPECT_EQ(dist[1], 0);
    EXPECT_EQ(dist[2], 10);
    EXPECT_EQ(dist[3], 5);  // 1->2->3 = 10 + (-5) = 5, shorter than 1->3 = 7

    std::cout << "dist[] = { " << dist[1] << " " << dist[2] << " " << dist[3]
              << " } -- negative edge correctly relaxed." << std::endl;
}

// ---------------------------------------------------------------------------
// Bellman-Ford detects negative cycle
// ---------------------------------------------------------------------------

TEST(BellmanFordTest, DetectsNegativeCycle) {
    PrintSection("Bellman-Ford Negative Cycle Detection",
                 "When a negative-weight cycle is reachable from the source, "
                 "BellmanFord must return an empty vector.");

    // 1 -> 2 (1), 2 -> 3 (1), 3 -> 1 (-5)  => total cycle weight = -3
    Graph<int> g(3);
    g.AddEdge(1, 2, 1,  false);
    g.AddEdge(2, 3, 1,  false);
    g.AddEdge(3, 1, -5, false);

    auto dist = g.BellmanFord(1);
    EXPECT_TRUE(dist.empty()) << "Expected negative cycle detection.";

    std::cout << "Negative cycle detected -- BellmanFord returned empty vector."
              << std::endl;
}

// ---------------------------------------------------------------------------
// Directed graph -- asymmetric distances
// ---------------------------------------------------------------------------

TEST(DijkstraTest, DirectedGraphAsymmetricDistances) {
    PrintSection("Directed Graph -- Asymmetric Distances",
                 "In a directed graph, reachability is one-way; the reverse "
                 "direction must show kInfinity for unreachable vertices.");

    Graph<int> g(3);
    g.AddEdge(1, 2, 1, false);
    g.AddEdge(2, 3, 2, false);

    auto from1 = g.Dijkstra(1);
    EXPECT_EQ(from1[1], 0);
    EXPECT_EQ(from1[2], 1);
    EXPECT_EQ(from1[3], 3);

    auto from3 = g.Dijkstra(3);
    EXPECT_EQ(from3[3], 0);
    EXPECT_EQ(from3[1], Graph<int>::kInfinity);
    EXPECT_EQ(from3[2], Graph<int>::kInfinity);

    std::cout << "From 1: { " << from1[1] << " " << from1[2] << " "
              << from1[3] << " }\n"
              << "From 3: { " << from3[1] << " " << from3[2] << " "
              << from3[3] << " } (unreachable = "
              << Graph<int>::kInfinity << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// Double-weight specialization
// ---------------------------------------------------------------------------

TEST(DijkstraTest, DoubleWeightType) {
    PrintSection("Double Weight Type",
                 "The template should work with floating-point weights, using "
                 "std::numeric_limits<double>::max() as infinity.");

    Graph<double> g(3);
    g.AddEdge(1, 2, 1.5);
    g.AddEdge(2, 3, 2.5);

    auto dist = g.Dijkstra(1);
    EXPECT_DOUBLE_EQ(dist[1], 0.0);
    EXPECT_DOUBLE_EQ(dist[2], 1.5);
    EXPECT_DOUBLE_EQ(dist[3], 4.0);
    EXPECT_EQ(Graph<double>::kInfinity, std::numeric_limits<double>::max());

    std::cout << "dist[] = { " << dist[1] << " " << dist[2] << " " << dist[3]
              << " } with Weight = double" << std::endl;
}

// ---------------------------------------------------------------------------
// Single vertex graph
// ---------------------------------------------------------------------------

TEST(DijkstraTest, SingleVertexGraph) {
    PrintSection("Single Vertex Graph",
                 "A graph with one vertex and no edges must return zero "
                 "distance for the source.");

    Graph<int> g(1);
    auto dist = g.Dijkstra(1);
    EXPECT_EQ(dist[1], 0);

    std::cout << "Single vertex dist[1] = " << dist[1] << std::endl;
}
