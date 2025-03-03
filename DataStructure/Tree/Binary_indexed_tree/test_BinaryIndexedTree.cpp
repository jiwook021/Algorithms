/**
 * @file test_BinaryIndexedTree.cpp
 * @brief Unit tests for BinaryIndexedTree (Fenwick Tree)
 */

#include "BinaryIndexedTree.hpp"
#include <gtest/gtest.h>

TEST(BinaryIndexedTreeTest, Construction) {
    EXPECT_NO_THROW(BinaryIndexedTree(10));
    EXPECT_THROW(BinaryIndexedTree(0), std::invalid_argument);
}

TEST(BinaryIndexedTreeTest, PrefixSum) {
    std::vector<long long> v = {1, 2, 3, 4, 5};
    BinaryIndexedTree bit(v);
    EXPECT_EQ(bit.PrefixSum(1), 1);
    EXPECT_EQ(bit.PrefixSum(3), 6);   // 1+2+3
    EXPECT_EQ(bit.PrefixSum(5), 15);  // 1+2+3+4+5
}

TEST(BinaryIndexedTreeTest, RangeSum) {
    std::vector<long long> v = {1, 2, 3, 4, 5};
    BinaryIndexedTree bit(v);
    EXPECT_EQ(bit.RangeSum(2, 4), 9);   // 2+3+4
    EXPECT_EQ(bit.RangeSum(1, 5), 15);
    EXPECT_EQ(bit.RangeSum(3, 3), 3);
}

TEST(BinaryIndexedTreeTest, PointUpdate) {
    std::vector<long long> v = {1, 2, 3, 4, 5};
    BinaryIndexedTree bit(v);
    bit.Update(3, 7);  // add 7 to index 3: 3 -> 10
    EXPECT_EQ(bit.PrefixSum(3), 13);  // 1+2+10
    EXPECT_EQ(bit.RangeSum(3, 3), 10);
}

TEST(BinaryIndexedTreeTest, OutOfRange) {
    BinaryIndexedTree bit(5);
    EXPECT_THROW(bit.PrefixSum(0), std::out_of_range);
    EXPECT_THROW(bit.PrefixSum(6), std::out_of_range);
    EXPECT_THROW(bit.Update(0, 1), std::out_of_range);
}

TEST(BinaryIndexedTreeTest, InvalidRange) {
    BinaryIndexedTree bit(5);
    EXPECT_THROW(bit.RangeSum(4, 2), std::invalid_argument);
}

TEST(BinaryIndexedTreeTest, LargeValues) {
    std::vector<long long> v(100, 1000000LL);
    BinaryIndexedTree bit(v);
    EXPECT_EQ(bit.PrefixSum(100), 100000000LL);
    EXPECT_EQ(bit.RangeSum(50, 60), 11000000LL);
}
