/**
 * @file test_DisjointSet.cpp
 * @brief Unit tests for DisjointSet (Union-Find)
 */

#include "DisjointSet.hpp"
#include <gtest/gtest.h>

/// Each element starts in its own set.
TEST(DisjointSetTest, InitiallyDisconnected) {
    DisjointSet ds(10);
    EXPECT_FALSE(ds.Connected(0, 1));
    EXPECT_FALSE(ds.Connected(2, 5));
    EXPECT_EQ(ds.Size(), 10u);
}

/// Self-find returns the element itself.
TEST(DisjointSetTest, SelfFind) {
    DisjointSet ds(5);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(ds.Find(i), i);
    }
}

/// Union merges two sets; Connected reflects the merge.
TEST(DisjointSetTest, UnionAndConnected) {
    DisjointSet ds(10);
    EXPECT_TRUE(ds.Union(1, 2));
    EXPECT_TRUE(ds.Connected(1, 2));
    EXPECT_FALSE(ds.Connected(1, 3));
}

/// Transitive connectivity: if 1-2 and 2-3 are unioned, 1-3 are connected.
TEST(DisjointSetTest, TransitiveUnion) {
    DisjointSet ds(10);
    ds.Union(1, 2);
    ds.Union(2, 3);
    EXPECT_TRUE(ds.Connected(1, 3));
}

/// Union returns false when elements are already in the same set.
TEST(DisjointSetTest, RedundantUnion) {
    DisjointSet ds(5);
    EXPECT_TRUE(ds.Union(0, 1));
    EXPECT_FALSE(ds.Union(0, 1));
}

/// Path compression: after Find, parent points directly to root.
TEST(DisjointSetTest, PathCompression) {
    DisjointSet ds(10);
    ds.Union(0, 1);
    ds.Union(1, 2);
    ds.Union(2, 3);
    // Find(3) compresses the path so all point to the same root.
    std::size_t root = ds.Find(3);
    EXPECT_EQ(ds.Find(0), root);
    EXPECT_EQ(ds.Find(1), root);
    EXPECT_EQ(ds.Find(2), root);
}

/// Two disjoint groups remain separate until explicitly merged.
TEST(DisjointSetTest, TwoGroups) {
    DisjointSet ds(10);
    ds.Union(1, 2);
    ds.Union(2, 3);
    ds.Union(5, 6);
    ds.Union(6, 7);
    EXPECT_FALSE(ds.Connected(1, 5));
    ds.Union(3, 5);
    EXPECT_TRUE(ds.Connected(1, 7));
}

/// Out-of-range index throws.
TEST(DisjointSetTest, OutOfRangeThrows) {
    DisjointSet ds(5);
    EXPECT_THROW(ds.Find(5), std::out_of_range);
    EXPECT_THROW(ds.Union(0, 10), std::out_of_range);
    EXPECT_THROW(ds.Connected(10, 0), std::out_of_range);
}

/// Zero-size construction throws.
TEST(DisjointSetTest, ZeroSizeThrows) {
    EXPECT_THROW(DisjointSet(0), std::invalid_argument);
}

/// Large-scale stress: union a chain of 1000 elements.
TEST(DisjointSetTest, LargeChain) {
    DisjointSet ds(1000);
    for (std::size_t i = 0; i + 1 < 1000; ++i) {
        ds.Union(i, i + 1);
    }
    EXPECT_TRUE(ds.Connected(0, 999));
}
