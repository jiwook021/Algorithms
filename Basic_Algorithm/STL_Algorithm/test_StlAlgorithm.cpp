/**
 * @file test_StlAlgorithm.cpp
 * @brief Unit tests for StlAlgorithm
 */

#include "StlAlgorithm.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace StlAlgorithm;

static auto isEven = [](int x) { return x % 2 == 0; };
static auto isNeg = [](int x) { return x < 0; };

TEST(StlAlgorithm, AnyOfFindsMatch) {
    std::vector<int> v = {1, 3, 5, 6};
    EXPECT_TRUE(MyAnyOf(v.begin(), v.end(), isEven));
}

TEST(StlAlgorithm, AnyOfNoMatch) {
    std::vector<int> v = {1, 3, 5};
    EXPECT_FALSE(MyAnyOf(v.begin(), v.end(), isEven));
}

TEST(StlAlgorithm, NoneOfAllMiss) {
    std::vector<int> v = {1, 3, 5};
    EXPECT_TRUE(MyNoneOf(v.begin(), v.end(), isEven));
}

TEST(StlAlgorithm, NoneOfHasMatch) {
    std::vector<int> v = {1, 3, 5, 2};
    EXPECT_FALSE(MyNoneOf(v.begin(), v.end(), isEven));
}

TEST(StlAlgorithm, FindIfFound) {
    std::vector<int> v = {1, 3, 4, 5};
    auto it = MyFindIf(v.begin(), v.end(), isEven);
    EXPECT_NE(it, v.end());
    EXPECT_EQ(*it, 4);
}

TEST(StlAlgorithm, FindIfNotFound) {
    std::vector<int> v = {1, 3, 5};
    auto it = MyFindIf(v.begin(), v.end(), isEven);
    EXPECT_EQ(it, v.end());
}

TEST(StlAlgorithm, FindIfNotBasic) {
    std::vector<int> v = {2, 4, 5, 6};
    auto it = MyFindIfNot(v.begin(), v.end(), isEven);
    EXPECT_NE(it, v.end());
    EXPECT_EQ(*it, 5);
}

TEST(StlAlgorithm, CountIf) {
    std::vector<int> v = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(MyCountIf(v.begin(), v.end(), isEven), 3);
}

TEST(StlAlgorithm, RemoveIf) {
    std::vector<int> v = {1, 2, 3, 4, 5};
    auto newEnd = MyRemoveIf(v.begin(), v.end(), isEven);
    std::vector<int> result(v.begin(), newEnd);
    EXPECT_EQ(result, (std::vector<int>{1, 3, 5}));
}

TEST(StlAlgorithm, IsPartitionedTrue) {
    std::vector<int> v = {2, 4, 1, 3};
    EXPECT_TRUE(MyIsPartitioned(v.begin(), v.end(), isEven));
}

TEST(StlAlgorithm, IsPartitionedFalse) {
    std::vector<int> v = {2, 1, 4, 3};
    EXPECT_FALSE(MyIsPartitioned(v.begin(), v.end(), isEven));
}

TEST(StlAlgorithm, EmptyRange) {
    std::vector<int> v;
    EXPECT_FALSE(MyAnyOf(v.begin(), v.end(), isEven));
    EXPECT_TRUE(MyNoneOf(v.begin(), v.end(), isEven));
    EXPECT_EQ(MyCountIf(v.begin(), v.end(), isEven), 0);
    EXPECT_TRUE(MyIsPartitioned(v.begin(), v.end(), isEven));
}
