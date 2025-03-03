/**
 * @file test_SlidingWindows.cpp
 * @brief Unit tests for SlidingWindows
 */

#include "SlidingWindows.hpp"
#include <gtest/gtest.h>

using namespace SlidingWindows;

TEST(SlidingWindows, BasicCase) {
    EXPECT_EQ(MaxSumSubarray({1, 2, 3, 4, 5}, 3), 12);
}

TEST(SlidingWindows, SingleElement) {
    EXPECT_EQ(MaxSumSubarray({5}, 1), 5);
}

TEST(SlidingWindows, WindowLargerThanArray) {
    EXPECT_EQ(MaxSumSubarray({1, 2}, 5), -1);
}

TEST(SlidingWindows, NegativeValues) {
    EXPECT_EQ(MaxSumSubarray({-1, -2, -3, -4}, 2), -3);
}

TEST(SlidingWindows, MixedValues) {
    EXPECT_EQ(MaxSumSubarray({1, 4, 2, 10, 2, 3, 1, 0, 20}, 4), 24);
}

TEST(SlidingWindows, AllSame) {
    EXPECT_EQ(MaxSumSubarray({3, 3, 3, 3}, 2), 6);
}

TEST(SlidingWindows, WindowEqualsArraySize) {
    EXPECT_EQ(MaxSumSubarray({1, 2, 3}, 3), 6);
}

TEST(SlidingWindows, EmptyArray) {
    EXPECT_EQ(MaxSumSubarray({}, 1), -1);
}
