/**
 * @file test_MergeSortThread.cpp
 * @brief Unit tests for MergeSortThread
 */

#include "MergeSortThread.hpp"
#include <gtest/gtest.h>
#include <random>

using namespace MergeSortThread;

TEST(MergeSortThread, BasicSort) {
    std::vector<int> v = {5, 3, 1, 4, 2};
    MySort(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(MergeSortThread, AlreadySorted) {
    std::vector<int> v = {1, 2, 3, 4, 5};
    MySort(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(MergeSortThread, ReverseSorted) {
    std::vector<int> v = {5, 4, 3, 2, 1};
    MySort(v.begin(), v.end());
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(MergeSortThread, Empty) {
    std::vector<int> v;
    MySort(v.begin(), v.end());
    EXPECT_TRUE(v.empty());
}

TEST(MergeSortThread, SingleElement) {
    std::vector<int> v = {42};
    MySort(v.begin(), v.end());
    EXPECT_EQ(v[0], 42);
}

TEST(MergeSortThread, Descending) {
    std::vector<int> v = {1, 3, 5, 2, 4};
    MySort(v.begin(), v.end(), std::greater<int>{});
    EXPECT_EQ(v, (std::vector<int>{5, 4, 3, 2, 1}));
}

TEST(MergeSortThread, Duplicates) {
    std::vector<int> v = {3, 1, 3, 2, 1};
    MySort(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 1, 2, 3, 3}));
}

TEST(MergeSortThread, LargeArray) {
    std::vector<int> v(50000);
    std::mt19937 gen(42);
    std::generate(v.begin(), v.end(), [&]() { return gen() % 100000; });
    MySort(v.begin(), v.end());
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}
