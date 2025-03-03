/**
 * @file test_SegmentTree.cpp
 * @brief Google Test suite for SegmentTree<T>.
 *
 * Covers: range sum, point update propagation, range minimum (custom op),
 *         range maximum, range GCD, single-element queries, and edge cases.
 */

#include <gtest/gtest.h>

#include "SegmentTree.hpp"

#include <algorithm>
#include <climits>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// PrintSection helper -- educational banners before each logical group
// ---------------------------------------------------------------------------

namespace {
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}
}  // namespace

// ===========================================================================
// Range-sum tests (default Op = std::plus<int>)
// ===========================================================================

TEST(SegmentTreeTest, BuildAndRangeSum) {
    PrintSection("BuildAndRangeSum",
                 "Verify that the tree correctly aggregates sums after Build.");

    std::vector<int> data = {1, 3, 5, 7};
    SegmentTree<int> st(data);

    // Total sum
    EXPECT_EQ(st.Query(0, 3), 16);
    // Partial sums
    EXPECT_EQ(st.Query(0, 1), 4);
    EXPECT_EQ(st.Query(2, 3), 12);
    EXPECT_EQ(st.Query(1, 2), 8);
}

TEST(SegmentTreeTest, SingleElementQuery) {
    PrintSection("SingleElementQuery",
                 "Each leaf should return its own value when queried alone.");

    std::vector<int> data = {1, 3, 5, 7};
    SegmentTree<int> st(data);

    EXPECT_EQ(st.Query(0, 0), 1);
    EXPECT_EQ(st.Query(1, 1), 3);
    EXPECT_EQ(st.Query(2, 2), 5);
    EXPECT_EQ(st.Query(3, 3), 7);
}

TEST(SegmentTreeTest, PointUpdatePropagation) {
    PrintSection("PointUpdatePropagation",
                 "After a point update every ancestor must reflect the change.");

    std::vector<int> data = {1, 3, 5, 7};
    SegmentTree<int> st(data);

    // Update index 3 (0-based) from 7 -> 3
    st.Update(3, 3);
    // New logical array: {1, 3, 5, 3}
    EXPECT_EQ(st.Query(0, 3), 12);
    EXPECT_EQ(st.Query(3, 3), 3);
    EXPECT_EQ(st.Query(1, 2), 8);  // unchanged region
}

TEST(SegmentTreeTest, MultipleUpdates) {
    PrintSection("MultipleUpdates",
                 "Consecutive updates must all propagate correctly.");

    std::vector<int> data = {2, 4, 6, 8, 10};
    SegmentTree<int> st(data);

    EXPECT_EQ(st.Query(0, 4), 30);

    st.Update(0, 0);   // {0, 4, 6, 8, 10}
    EXPECT_EQ(st.Query(0, 4), 28);

    st.Update(2, 1);   // {0, 4, 1, 8, 10}
    EXPECT_EQ(st.Query(0, 4), 23);
    EXPECT_EQ(st.Query(1, 3), 13);
}

TEST(SegmentTreeTest, SizeAccessor) {
    PrintSection("SizeAccessor",
                 "Size() returns the power-of-2 capacity, always >= n.");

    std::vector<int> data = {1, 2, 3};
    SegmentTree<int> st(data);

    EXPECT_GE(st.Size(), data.size());
}

// ===========================================================================
// Range-minimum tests (custom Op)
// ===========================================================================

TEST(SegmentTreeTest, RangeMinimum) {
    PrintSection("RangeMinimum",
                 "SegmentTree with std::min should return the minimum in any range.");

    SegmentTree<int> st({5, 2, 8, 1, 9, 3}, INT_MAX,
                        [](int a, int b) { return std::min(a, b); });

    EXPECT_EQ(st.Query(0, 5), 1);   // global min
    EXPECT_EQ(st.Query(0, 2), 2);   // min of {5,2,8}
    EXPECT_EQ(st.Query(3, 5), 1);   // min of {1,9,3}
    EXPECT_EQ(st.Query(4, 5), 3);   // min of {9,3}
}

TEST(SegmentTreeTest, RangeMinimumAfterUpdate) {
    PrintSection("RangeMinimumAfterUpdate",
                 "Point update should change which element is the minimum.");

    SegmentTree<int> st({5, 2, 8, 1, 9, 3}, INT_MAX,
                        [](int a, int b) { return std::min(a, b); });

    st.Update(3, 20);  // change the 1 -> 20; array becomes {5,2,8,20,9,3}
    EXPECT_EQ(st.Query(0, 5), 2);   // new global min
    EXPECT_EQ(st.Query(3, 5), 3);   // min of {20,9,3}
}

// ===========================================================================
// Range-maximum tests
// ===========================================================================

TEST(SegmentTreeTest, RangeMaximum) {
    PrintSection("RangeMaximum",
                 "SegmentTree with std::max should return the maximum in any range.");

    SegmentTree<int> st({5, 2, 8, 1, 9, 3}, INT_MIN,
                        [](int a, int b) { return std::max(a, b); });

    EXPECT_EQ(st.Query(0, 5), 9);   // global max
    EXPECT_EQ(st.Query(0, 2), 8);   // max of {5,2,8}
    EXPECT_EQ(st.Query(3, 5), 9);   // max of {1,9,3}
}

// ===========================================================================
// Range-GCD tests
// ===========================================================================

TEST(SegmentTreeTest, RangeGCD) {
    PrintSection("RangeGCD",
                 "GCD is associative; the tree should compute it over any sub-range.");

    SegmentTree<int> st({12, 18, 24, 36}, 0,
                        [](int a, int b) { return std::gcd(a, b); });

    EXPECT_EQ(st.Query(0, 3), 6);    // gcd(12,18,24,36) = 6
    EXPECT_EQ(st.Query(0, 1), 6);    // gcd(12,18) = 6
    EXPECT_EQ(st.Query(2, 3), 12);   // gcd(24,36) = 12
    EXPECT_EQ(st.Query(1, 2), 6);    // gcd(18,24) = 6
}

TEST(SegmentTreeTest, RangeGCDAfterUpdate) {
    PrintSection("RangeGCDAfterUpdate",
                 "Updating an element should change the GCD over ranges that include it.");

    SegmentTree<int> st({12, 18, 24, 36}, 0,
                        [](int a, int b) { return std::gcd(a, b); });

    st.Update(1, 12);  // array becomes {12, 12, 24, 36}
    EXPECT_EQ(st.Query(0, 3), 12);   // gcd(12,12,24,36) = 12
    EXPECT_EQ(st.Query(0, 1), 12);   // gcd(12,12) = 12
}

// ===========================================================================
// Edge cases
// ===========================================================================

TEST(SegmentTreeTest, SingleElementTree) {
    PrintSection("SingleElementTree",
                 "A tree with exactly one element is a valid edge case.");

    SegmentTree<int> st({42});

    EXPECT_EQ(st.Query(0, 0), 42);
    st.Update(0, 99);
    EXPECT_EQ(st.Query(0, 0), 99);
}

TEST(SegmentTreeTest, LargeArray) {
    PrintSection("LargeArray",
                 "Stress-test with 10000 elements to verify correctness at scale.");

    const int kN = 10000;
    std::vector<int> data(kN);
    for (int i = 0; i < kN; ++i) data[i] = i + 1;

    SegmentTree<int> st(data);

    // Sum of [0..kN-1] = kN*(kN+1)/2
    long long expected = static_cast<long long>(kN) * (kN + 1) / 2;
    EXPECT_EQ(st.Query(0, kN - 1), static_cast<int>(expected));

    // Partial range: sum of [100..199] = sum(101..200)
    int partial = 0;
    for (int i = 100; i <= 199; ++i) partial += data[i];
    EXPECT_EQ(st.Query(100, 199), partial);
}
