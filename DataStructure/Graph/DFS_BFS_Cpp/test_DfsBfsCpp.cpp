/**
 * @file test_DfsBfsCpp.cpp
 * @brief Unit tests for DfsBfsCpp functional-style graph traversal
 */

#include "DfsBfsCpp.hpp"
#include <gtest/gtest.h>

TEST(DfsBfsCpp, AddGraphCreatesUndirectedEdge) {
    std::vector<std::vector<int>> g(5);
    AddGraph(0, 1, g);
    EXPECT_EQ(g[0].size(), 1u);
    EXPECT_EQ(g[1].size(), 1u);
    EXPECT_EQ(g[0][0], 1);
    EXPECT_EQ(g[1][0], 0);
}

TEST(DfsBfsCpp, DFSVisitsAllConnected) {
    std::vector<std::vector<int>> g(4);
    AddGraph(0, 1, g);
    AddGraph(1, 2, g);
    AddGraph(2, 3, g);
    auto order = DfsCollect(0, g);
    EXPECT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], 0);
}

TEST(DfsBfsCpp, BFSVisitsAllConnected) {
    std::vector<std::vector<int>> g(4);
    AddGraph(0, 1, g);
    AddGraph(0, 2, g);
    AddGraph(1, 3, g);
    auto order = BfsCollect(0, g);
    EXPECT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], 0);
}

TEST(DfsBfsCpp, DisconnectedGraph) {
    std::vector<std::vector<int>> g(4);
    AddGraph(0, 1, g);
    auto order = DfsCollect(0, g);
    EXPECT_EQ(order.size(), 2u);
}

TEST(DfsBfsCpp, SingleNode) {
    std::vector<std::vector<int>> g(1);
    auto dfs = DfsCollect(0, g);
    auto bfs = BfsCollect(0, g);
    EXPECT_EQ(dfs.size(), 1u);
    EXPECT_EQ(bfs.size(), 1u);
}
