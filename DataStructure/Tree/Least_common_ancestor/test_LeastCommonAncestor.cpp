/**
 * @file test_LeastCommonAncestor.cpp
 * @brief Unit tests for LeastCommonAncestor (Euler tour + Sparse Table RMQ)
 */

#include "LeastCommonAncestor.hpp"
#include <gtest/gtest.h>

class LcaTest : public ::testing::Test {
protected:
    // Tree:
    //        0
    //       / |
    //      1   2
    //     / |   |
    //    3   4   5
    //   /
    //  6
    LeastCommonAncestor lca{7};

    void SetUp() override {
        lca.AddEdge(0, 1);
        lca.AddEdge(0, 2);
        lca.AddEdge(1, 3);
        lca.AddEdge(1, 4);
        lca.AddEdge(2, 5);
        lca.AddEdge(3, 6);
        lca.Build(0);
    }
};

TEST_F(LcaTest, SiblingsShareParent) {
    EXPECT_EQ(lca.Query(3, 4), 1u);  // siblings under node 1
}

TEST_F(LcaTest, ParentAndChild) {
    EXPECT_EQ(lca.Query(1, 3), 1u);  // parent-child
    EXPECT_EQ(lca.Query(0, 6), 0u);  // root and descendant
}

TEST_F(LcaTest, SameNode) {
    EXPECT_EQ(lca.Query(3, 3), 3u);
    EXPECT_EQ(lca.Query(0, 0), 0u);
}

TEST_F(LcaTest, CrossSubtrees) {
    EXPECT_EQ(lca.Query(6, 5), 0u);  // left subtree and right subtree
    EXPECT_EQ(lca.Query(3, 5), 0u);
    EXPECT_EQ(lca.Query(4, 5), 0u);
}

TEST_F(LcaTest, DescendantAndUncle) {
    EXPECT_EQ(lca.Query(6, 4), 1u);  // 6 is under 3, 4 is sibling of 3
}

TEST_F(LcaTest, QueryOrderIndependent) {
    EXPECT_EQ(lca.Query(3, 5), lca.Query(5, 3));
    EXPECT_EQ(lca.Query(6, 2), lca.Query(2, 6));
}

TEST(LcaConstructionTest, InvalidNodeCount) {
    EXPECT_THROW(LeastCommonAncestor(0), std::invalid_argument);
}

TEST(LcaConstructionTest, QueryBeforeBuild) {
    LeastCommonAncestor lca(3);
    lca.AddEdge(0, 1);
    lca.AddEdge(0, 2);
    EXPECT_THROW(lca.Query(1, 2), std::logic_error);
}

TEST(LcaConstructionTest, OutOfRangeNode) {
    LeastCommonAncestor lca(3);
    EXPECT_THROW(lca.AddEdge(0, 5), std::out_of_range);
}

TEST(LcaConstructionTest, SingleNode) {
    LeastCommonAncestor lca(1);
    lca.Build(0);
    EXPECT_EQ(lca.Query(0, 0), 0u);
}

TEST(LcaConstructionTest, LinearChain) {
    // 0 - 1 - 2 - 3 - 4
    LeastCommonAncestor lca(5);
    lca.AddEdge(0, 1);
    lca.AddEdge(1, 2);
    lca.AddEdge(2, 3);
    lca.AddEdge(3, 4);
    lca.Build(0);

    EXPECT_EQ(lca.Query(0, 4), 0u);
    EXPECT_EQ(lca.Query(2, 4), 2u);
    EXPECT_EQ(lca.Query(1, 3), 1u);
}

TEST(LcaConstructionTest, StarGraph) {
    // 0 connected to 1, 2, 3, 4
    LeastCommonAncestor lca(5);
    for (std::size_t i = 1; i < 5; ++i) {
        lca.AddEdge(0, i);
    }
    lca.Build(0);

    EXPECT_EQ(lca.Query(1, 2), 0u);
    EXPECT_EQ(lca.Query(3, 4), 0u);
    EXPECT_EQ(lca.Query(1, 4), 0u);
}
