/**
 * @file test_Iterators.cpp
 * @brief Unit tests for NumIterator and NumRange
 */

#include "Iterators.hpp"
#include <gtest/gtest.h>
#include <vector>

/// Dereferencing an iterator returns its position value.
TEST(NumIteratorTest, Dereference) {
    NumIterator it(5);
    EXPECT_EQ(*it, 5);
}

/// Pre-increment advances the iterator by one.
TEST(NumIteratorTest, Increment) {
    NumIterator it(5);
    ++it;
    EXPECT_EQ(*it, 6);
}

/// Inequality comparison works correctly.
TEST(NumIteratorTest, Inequality) {
    NumIterator a(3), b(5);
    EXPECT_TRUE(a != b);
    NumIterator c(3);
    EXPECT_FALSE(a != c);
}

/// NumRange produces the expected sequence of integers.
TEST(NumRangeTest, ProducesCorrectSequence) {
    std::vector<int> result;
    for (int i : NumRange(1, 4)) {
        result.push_back(i);
    }
    EXPECT_EQ(result, (std::vector<int>{1, 2, 3}));
}

/// NumRange sum over [1, 4) = 1 + 2 + 3 = 6.
TEST(NumRangeTest, Sum) {
    int sum = 0;
    for (int i : NumRange(1, 4)) sum += i;
    EXPECT_EQ(sum, 6);
}

/// Empty range (from == to) yields no iterations.
TEST(NumRangeTest, EmptyRange) {
    int count = 0;
    for ([[maybe_unused]] int i : NumRange(5, 5)) count++;
    EXPECT_EQ(count, 0);
}

/// Single-element range [10, 11) yields exactly one value.
TEST(NumRangeTest, SingleElement) {
    std::vector<int> result;
    for (int i : NumRange(10, 11)) result.push_back(i);
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 10);
}

/// Large range produces correct count.
TEST(NumRangeTest, LargeRange) {
    int count = 0;
    for ([[maybe_unused]] int i : NumRange(0, 1000)) count++;
    EXPECT_EQ(count, 1000);
}
