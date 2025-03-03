/**
 * @file test_KruskalAlgorithm.cpp
 * @brief Unit tests for Kruskal's MST algorithm.
 */

#include "KruskalAlgorithm.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <set>

// --- Construction ---

TEST(KruskalGraphTest, ConstructionValid) {
    EXPECT_NO_THROW(KruskalGraph(5));
    KruskalGraph g(3);
    EXPECT_EQ(g.NumVertices(), 3);
    EXPECT_EQ(g.NumEdges(), 0);
}

TEST(KruskalGraphTest, ConstructionInvalid) {
    EXPECT_THROW(KruskalGraph(0), std::invalid_argument);
    EXPECT_THROW(KruskalGraph(-1), std::invalid_argument);
}

// --- AddEdge ---

TEST(KruskalGraphTest, AddEdge) {
    KruskalGraph g(4);
    g.AddEdge(0, 1, 5);
    g.AddEdge(1, 2, 3);
    EXPECT_EQ(g.NumEdges(), 2);
}

TEST(KruskalGraphTest, AddEdgeOutOfRange) {
    KruskalGraph g(3);
    EXPECT_THROW(g.AddEdge(0, 5, 1), std::out_of_range);
    EXPECT_THROW(g.AddEdge(-1, 0, 1), std::out_of_range);
}

// --- MST on classic example ---

TEST(KruskalGraphTest, ClassicExampleMST) {
    // 6 vertices (A=0, B=1, C=2, D=3, E=4, F=5)
    KruskalGraph g(6);
    g.AddEdge(0, 1, 9);   // A-B
    g.AddEdge(1, 2, 2);   // B-C
    g.AddEdge(0, 2, 12);  // A-C
    g.AddEdge(0, 3, 8);   // A-D
    g.AddEdge(3, 2, 6);   // D-C
    g.AddEdge(0, 5, 11);  // A-F
    g.AddEdge(5, 3, 4);   // F-D
    g.AddEdge(3, 4, 3);   // D-E
    g.AddEdge(4, 2, 7);   // E-C
    g.AddEdge(5, 4, 13);  // F-E

    auto mst = g.ComputeMST();

    // MST should have V-1 = 5 edges
    EXPECT_EQ(mst.size(), 5u);

    // Total weight should be 2+3+4+6+8 = 23
    EXPECT_EQ(g.MSTWeight(), 23);

    // Check that MST edges are sorted by weight
    for (std::size_t i = 1; i < mst.size(); ++i) {
        EXPECT_LE(mst[i - 1].weight, mst[i].weight);
    }

    // Verify the expected edges by collecting weights
    std::vector<int> mstWeights;
    for (const Edge& e : mst) {
        mstWeights.push_back(e.weight);
    }
    std::sort(mstWeights.begin(), mstWeights.end());
    EXPECT_EQ(mstWeights, (std::vector<int>{2, 3, 4, 6, 8}));
}

// --- Simple triangle ---

TEST(KruskalGraphTest, TriangleMST) {
    KruskalGraph g(3);
    g.AddEdge(0, 1, 1);
    g.AddEdge(1, 2, 2);
    g.AddEdge(0, 2, 3);

    auto mst = g.ComputeMST();
    EXPECT_EQ(mst.size(), 2u);
    EXPECT_EQ(g.MSTWeight(), 3);  // 1 + 2

    // The lightest two edges should be selected
    EXPECT_EQ(mst[0].weight, 1);
    EXPECT_EQ(mst[1].weight, 2);
}

// --- Single vertex (no edges needed) ---

TEST(KruskalGraphTest, SingleVertexMST) {
    KruskalGraph g(1);
    auto mst = g.ComputeMST();
    EXPECT_EQ(mst.size(), 0u);
    EXPECT_EQ(g.MSTWeight(), 0);
}

// --- Two vertices, one edge ---

TEST(KruskalGraphTest, TwoVerticesMST) {
    KruskalGraph g(2);
    g.AddEdge(0, 1, 7);
    auto mst = g.ComputeMST();
    EXPECT_EQ(mst.size(), 1u);
    EXPECT_EQ(mst[0].weight, 7);
}

// --- Disconnected graph (spanning forest) ---

TEST(KruskalGraphTest, DisconnectedGraphMSF) {
    KruskalGraph g(4);
    g.AddEdge(0, 1, 1);
    g.AddEdge(2, 3, 2);
    // No edges connecting {0,1} to {2,3}

    auto mst = g.ComputeMST();
    EXPECT_EQ(mst.size(), 2u);  // Two separate components
    EXPECT_EQ(g.MSTWeight(), 3);
}

// --- Linear chain ---

TEST(KruskalGraphTest, LinearChainMST) {
    KruskalGraph g(5);
    g.AddEdge(0, 1, 1);
    g.AddEdge(1, 2, 2);
    g.AddEdge(2, 3, 3);
    g.AddEdge(3, 4, 4);

    auto mst = g.ComputeMST();
    EXPECT_EQ(mst.size(), 4u);
    EXPECT_EQ(g.MSTWeight(), 10);
}

// --- Duplicate weights ---

TEST(KruskalGraphTest, DuplicateWeightsMST) {
    KruskalGraph g(4);
    g.AddEdge(0, 1, 5);
    g.AddEdge(1, 2, 5);
    g.AddEdge(2, 3, 5);
    g.AddEdge(0, 3, 5);

    auto mst = g.ComputeMST();
    EXPECT_EQ(mst.size(), 3u);
    EXPECT_EQ(g.MSTWeight(), 15);
}

// --- DisjointSetKruskal unit tests ---

TEST(DisjointSetKruskalTest, InitiallyDisconnected) {
    DisjointSetKruskal ds(5);
    EXPECT_NE(ds.Find(0), ds.Find(1));
    EXPECT_NE(ds.Find(2), ds.Find(3));
}

TEST(DisjointSetKruskalTest, UnionAndFind) {
    DisjointSetKruskal ds(5);
    EXPECT_TRUE(ds.Union(0, 1));
    EXPECT_EQ(ds.Find(0), ds.Find(1));
    EXPECT_FALSE(ds.Union(0, 1));  // Already same set
}

TEST(DisjointSetKruskalTest, TransitiveUnion) {
    DisjointSetKruskal ds(5);
    ds.Union(0, 1);
    ds.Union(1, 2);
    EXPECT_EQ(ds.Find(0), ds.Find(2));
}
