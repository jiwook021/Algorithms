/**
 * @file test_GraphDfsBfsFull.cpp
 * @brief Google Test suite for GraphFull -- iterative DFS/BFS with std::span neighbors.
 *
 * Educational PrintSection output explains *why* each group of tests matters.
 */

#include "GraphDfsBfsFull.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

// Helper: build the standard 5-vertex test graph.
//   0--1, 0--3, 1--2, 2--3, 3--4, 4--0
GraphFull MakeTestGraph() {
    GraphFull g(5);
    g.AddEdge(0, 1);
    g.AddEdge(0, 3);
    g.AddEdge(1, 2);
    g.AddEdge(2, 3);
    g.AddEdge(3, 4);
    g.AddEdge(4, 0);
    return g;
}

} // namespace

// ===========================================================================
// Construction
// ===========================================================================

TEST(GraphFullTest, ConstructionValid) {
    PrintSection("Construction -- valid",
                 "Ensures the graph allocates the requested number of vertices "
                 "and starts with zero edges.");

    GraphFull g(5);
    EXPECT_EQ(g.NumVertices(), 5u);
    EXPECT_EQ(g.NumEdges(), 0u);
}

TEST(GraphFullTest, ConstructionZeroThrows) {
    PrintSection("Construction -- zero vertices",
                 "A graph with zero vertices is meaningless; the constructor "
                 "must reject it.");

    EXPECT_THROW(GraphFull(0), std::invalid_argument);
}

// ===========================================================================
// AddEdge
// ===========================================================================

TEST(GraphFullTest, AddEdgeCreatesUndirectedLink) {
    PrintSection("AddEdge -- undirected link",
                 "Each edge must appear in both endpoints' adjacency lists.");

    GraphFull g(4);
    g.AddEdge(0, 1);

    EXPECT_EQ(g.NumEdges(), 1u);
    ASSERT_EQ(g.Neighbors(0).size(), 1u);
    EXPECT_EQ(g.Neighbors(0)[0], 0u + 1u);
    ASSERT_EQ(g.Neighbors(1).size(), 1u);
    EXPECT_EQ(g.Neighbors(1)[0], 0u);
}

TEST(GraphFullTest, AddEdgeOutOfRange) {
    PrintSection("AddEdge -- out-of-range vertex",
                 "The graph must validate vertex indices to prevent UB.");

    GraphFull g(3);
    EXPECT_THROW(g.AddEdge(0, 5), std::out_of_range);
}

TEST(GraphFullTest, NeighborsSorted) {
    PrintSection("Neighbors -- sorted adjacency",
                 "Sorted neighbor lists make traversal order deterministic, "
                 "which is critical for reproducible BFS/DFS results.");

    GraphFull g(5);
    g.AddEdge(0, 3);
    g.AddEdge(0, 1);
    g.AddEdge(0, 4);
    g.AddEdge(0, 2);

    auto nbrs = g.Neighbors(0);
    std::vector<std::size_t> expected{1, 2, 3, 4};
    ASSERT_EQ(nbrs.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(nbrs[i], expected[i]);
    }
}

// ===========================================================================
// DFS -- depth-first search
// ===========================================================================

TEST(GraphFullTest, DfsVisitsAllReachable) {
    PrintSection("DFS -- visits all reachable vertices",
                 "DFS must explore every vertex connected to the start, "
                 "regardless of graph shape.");

    auto g = MakeTestGraph();
    auto order = g.DFS(0);

    EXPECT_EQ(order.size(), 5u);
    EXPECT_EQ(order.front(), 0u);

    // Every vertex 0..4 must appear exactly once.
    auto sorted = order;
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::size_t> all(5);
    std::iota(all.begin(), all.end(), 0u);
    EXPECT_EQ(sorted, all);
}

TEST(GraphFullTest, DfsSingleVertex) {
    PrintSection("DFS -- single vertex",
                 "Edge case: a graph with one vertex has a trivial DFS.");

    GraphFull g(1);
    auto order = g.DFS(0);
    ASSERT_EQ(order.size(), 1u);
    EXPECT_EQ(order[0], 0u);
}

TEST(GraphFullTest, DfsDisconnectedPartialTraversal) {
    PrintSection("DFS -- disconnected graph partial traversal",
                 "DFS from a vertex in a disconnected graph must only visit "
                 "the connected component containing the start vertex.");

    GraphFull g(6);
    // Component A: 0--1--2
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    // Component B: 3--4--5
    g.AddEdge(3, 4);
    g.AddEdge(4, 5);

    auto from_a = g.DFS(0);
    EXPECT_EQ(from_a.size(), 3u);
    // None of 3,4,5 should appear.
    for (std::size_t v : from_a) {
        EXPECT_LT(v, 3u);
    }

    auto from_b = g.DFS(3);
    EXPECT_EQ(from_b.size(), 3u);
    for (std::size_t v : from_b) {
        EXPECT_GE(v, 3u);
    }
}

TEST(GraphFullTest, DfsLinearChain) {
    PrintSection("DFS -- linear chain",
                 "On a simple path 0-1-2-3-4, DFS must visit vertices in "
                 "order because each vertex has only one unvisited neighbor.");

    GraphFull g(5);
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(2, 3);
    g.AddEdge(3, 4);

    auto order = g.DFS(0);
    std::vector<std::size_t> expected{0, 1, 2, 3, 4};
    EXPECT_EQ(order, expected);
}

TEST(GraphFullTest, DfsOutOfRange) {
    PrintSection("DFS -- out-of-range start",
                 "Starting DFS from an invalid vertex must throw.");

    GraphFull g(3);
    EXPECT_THROW(g.DFS(5), std::out_of_range);
}

// ===========================================================================
// BFS -- breadth-first search
// ===========================================================================

TEST(GraphFullTest, BfsLayerByLayer) {
    PrintSection("BFS -- layer-by-layer traversal",
                 "BFS must explore vertices in order of their distance from "
                 "the start vertex (level 0, then level 1, etc.).");

    auto g = MakeTestGraph();
    auto order = g.BFS(0);

    EXPECT_EQ(order.size(), 5u);
    EXPECT_EQ(order[0], 0u);  // level 0

    // Neighbors of 0 are {1, 3, 4} (sorted). They form level 1.
    EXPECT_EQ(order[1], 1u);
    EXPECT_EQ(order[2], 3u);
    EXPECT_EQ(order[3], 4u);

    // Vertex 2 is at level 2 (reached through 1 or 3).
    EXPECT_EQ(order[4], 2u);
}

TEST(GraphFullTest, BfsSingleVertex) {
    PrintSection("BFS -- single vertex",
                 "Edge case: a graph with one vertex has a trivial BFS.");

    GraphFull g(1);
    auto order = g.BFS(0);
    ASSERT_EQ(order.size(), 1u);
    EXPECT_EQ(order[0], 0u);
}

TEST(GraphFullTest, BfsDisconnectedPartialTraversal) {
    PrintSection("BFS -- disconnected graph partial traversal",
                 "BFS from a vertex in a disconnected graph must only visit "
                 "the connected component containing the start vertex.");

    GraphFull g(6);
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(3, 4);
    g.AddEdge(4, 5);

    auto from_a = g.BFS(0);
    EXPECT_EQ(from_a.size(), 3u);
    for (std::size_t v : from_a) {
        EXPECT_LT(v, 3u);
    }
}

TEST(GraphFullTest, BfsLinearChain) {
    PrintSection("BFS -- linear chain",
                 "On a simple path, BFS and DFS produce the same order.");

    GraphFull g(5);
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(2, 3);
    g.AddEdge(3, 4);

    auto order = g.BFS(0);
    std::vector<std::size_t> expected{0, 1, 2, 3, 4};
    EXPECT_EQ(order, expected);
}

TEST(GraphFullTest, BfsOutOfRange) {
    PrintSection("BFS -- out-of-range start",
                 "Starting BFS from an invalid vertex must throw.");

    GraphFull g(3);
    EXPECT_THROW(g.BFS(5), std::out_of_range);
}

// ===========================================================================
// std::span -- neighbor access
// ===========================================================================

TEST(GraphFullTest, SpanNeighborAccessZeroCopy) {
    PrintSection("std::span -- zero-copy neighbor access",
                 "Neighbors() returns a std::span<const std::size_t>, "
                 "providing a lightweight, non-owning view into the internal "
                 "adjacency vector without any heap allocation.");

    auto g = MakeTestGraph();

    // Vertex 0 is connected to 1, 3, 4 (sorted).
    auto span0 = g.Neighbors(0);
    std::vector<std::size_t> expected0{1, 3, 4};
    ASSERT_EQ(span0.size(), expected0.size());
    for (std::size_t i = 0; i < expected0.size(); ++i) {
        EXPECT_EQ(span0[i], expected0[i]);
    }

    // Verify range-for works on the span.
    std::vector<std::size_t> collected;
    for (std::size_t n : span0) {
        collected.push_back(n);
    }
    EXPECT_EQ(collected, expected0);
}

TEST(GraphFullTest, SpanNeighborIsolatedVertex) {
    PrintSection("std::span -- isolated vertex returns empty span",
                 "An isolated vertex has no neighbors; the span must be empty.");

    GraphFull g(3);
    g.AddEdge(0, 1);

    auto span2 = g.Neighbors(2);
    EXPECT_TRUE(span2.empty());
    EXPECT_EQ(span2.size(), 0u);
}

TEST(GraphFullTest, SpanNeighborOutOfRange) {
    PrintSection("std::span -- out-of-range vertex",
                 "Requesting neighbors of a non-existent vertex must throw.");

    GraphFull g(3);
    EXPECT_THROW((void)g.Neighbors(10), std::out_of_range);
}

// ===========================================================================
// Cycle handling
// ===========================================================================

TEST(GraphFullTest, CyclicGraphTraversal) {
    PrintSection("Cycle handling",
                 "Both DFS and BFS must correctly handle cycles without "
                 "infinite loops, visiting each vertex exactly once.");

    GraphFull g(3);
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(2, 0);

    auto dfs = g.DFS(0);
    EXPECT_EQ(dfs.size(), 3u);
    auto sorted = dfs;
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::size_t> all{0, 1, 2};
    EXPECT_EQ(sorted, all);

    auto bfs = g.BFS(0);
    EXPECT_EQ(bfs.size(), 3u);
}
