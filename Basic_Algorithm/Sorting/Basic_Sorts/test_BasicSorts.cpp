/**
 * @file test_BasicSorts.cpp
 * @brief Unit tests for BasicSorts (bubble, insertion, selection)
 */

#include "BasicSorts.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <functional>

using namespace BasicSorts;

using SortFn = void (*)(std::vector<int>::iterator, std::vector<int>::iterator);

class BasicSortTest : public ::testing::TestWithParam<SortFn> {};

TEST_P(BasicSortTest, EmptyRange) {
    std::vector<int> v;
    GetParam()(v.begin(), v.end());
    EXPECT_TRUE(v.empty());
}

TEST_P(BasicSortTest, SingleElement) {
    std::vector<int> v{42};
    GetParam()(v.begin(), v.end());
    EXPECT_EQ(v, std::vector<int>{42});
}

TEST_P(BasicSortTest, AlreadySorted) {
    std::vector<int> v{1, 2, 3, 4, 5};
    GetParam()(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST_P(BasicSortTest, ReverseSorted) {
    std::vector<int> v{5, 4, 3, 2, 1};
    GetParam()(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST_P(BasicSortTest, RandomOrder) {
    std::vector<int> v{3, 1, 4, 1, 5, 9, 2, 6};
    GetParam()(v.begin(), v.end());
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST_P(BasicSortTest, Duplicates) {
    std::vector<int> v{5, 3, 3, 1, 5, 2, 1};
    GetParam()(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 1, 2, 3, 3, 5, 5}));
}

TEST_P(BasicSortTest, AllEqual) {
    std::vector<int> v(8, 7);
    GetParam()(v.begin(), v.end());
    EXPECT_EQ(v, std::vector<int>(8, 7));
}

TEST_P(BasicSortTest, TwoElements) {
    std::vector<int> v{9, 1};
    GetParam()(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<int>{1, 9}));
}

INSTANTIATE_TEST_SUITE_P(
    BubbleSort, BasicSortTest,
    ::testing::Values(static_cast<SortFn>(BubbleSort)));

INSTANTIATE_TEST_SUITE_P(
    InsertionSort, BasicSortTest,
    ::testing::Values(static_cast<SortFn>(InsertionSort)));

INSTANTIATE_TEST_SUITE_P(
    SelectionSort, BasicSortTest,
    ::testing::Values(static_cast<SortFn>(SelectionSort)));

TEST(BasicSortExtra, BubbleSortStrings) {
    std::vector<std::string> v{"banana", "apple", "cherry"};
    BubbleSort(v.begin(), v.end());
    EXPECT_EQ(v, (std::vector<std::string>{"apple", "banana", "cherry"}));
}

TEST(BasicSortExtra, InsertionSortDescending) {
    std::vector<int> v{1, 2, 3, 4, 5};
    InsertionSort(v.begin(), v.end(), std::greater<int>{});
    EXPECT_EQ(v, (std::vector<int>{5, 4, 3, 2, 1}));
}
