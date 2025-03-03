/**
 * @file test_DFS_BFS.cpp
 * @brief Unit tests for Graph DFS/BFS traversal
 */

#include "DFS_BFS.hpp"
#include <gtest/gtest.h>
#include <algorithm>

TEST(GraphDFSBFS, Construction) {
    EXPECT_NO_THROW(Graph(5));
    EXPECT_THROW(Graph(0), std::invalid_argument);
    EXPECT_THROW(Graph(-1), std::invalid_argument);
}

TEST(GraphDFSBFS, AddEdgeValid) {
    Graph g(5);
    EXPECT_NO_THROW(g.AddEdge(0, 1));
    EXPECT_NO_THROW(g.AddEdge(0, 4));
}

TEST(GraphDFSBFS, AddEdgeInvalid) {
    Graph g(5);
    EXPECT_THROW(g.AddEdge(0, 5), std::out_of_range);
    EXPECT_THROW(g.AddEdge(-1, 0), std::out_of_range);
}

TEST(GraphDFSBFS, DFSVisitsAllConnected) {
    Graph g(4);
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(2, 3);
    auto order = g.DFS(0);
    EXPECT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], 0);
}

TEST(GraphDFSBFS, BFSVisitsAllConnected) {
    Graph g(4);
    g.AddEdge(0, 1);
    g.AddEdge(0, 2);
    g.AddEdge(1, 3);
    auto order = g.BFS(0);
    EXPECT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], 0);
    // BFS: neighbors of 0 come before neighbors of 1
    EXPECT_TRUE(order[1] == 1 || order[1] == 2);
}

TEST(GraphDFSBFS, DisconnectedGraph) {
    Graph g(4);
    g.AddEdge(0, 1);
    // 2 and 3 are isolated
    auto order = g.DFS(0);
    EXPECT_EQ(order.size(), 2u);
}

TEST(GraphDFSBFS, SingleNode) {
    Graph g(1);
    auto dfs = g.DFS(0);
    auto bfs = g.BFS(0);
    EXPECT_EQ(dfs.size(), 1u);
    EXPECT_EQ(bfs.size(), 1u);
    EXPECT_EQ(dfs[0], 0);
    EXPECT_EQ(bfs[0], 0);
}

TEST(GraphDFSBFS, InvalidStartNode) {
    Graph g(3);
    EXPECT_THROW(g.DFS(5), std::invalid_argument);
    EXPECT_THROW(g.BFS(-1), std::invalid_argument);
}

TEST(GraphDFSBFS, CyclicGraph) {
    Graph g(3);
    g.AddEdge(0, 1);
    g.AddEdge(1, 2);
    g.AddEdge(2, 0);
    auto order = g.DFS(0);
    EXPECT_EQ(order.size(), 3u);
    // All nodes visited exactly once despite cycle
    std::sort(order.begin(), order.end());
    EXPECT_EQ(order, (std::vector<int>{0, 1, 2}));
}
