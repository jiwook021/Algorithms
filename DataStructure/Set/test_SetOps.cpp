/**
 * @file test_SetOps.cpp
 * @brief Unit tests for SetOps (set union, intersection, difference)
 */

#include "SetOps.hpp"
#include <gtest/gtest.h>
#include <string>

// --- Union Tests ---

TEST(SetOpsTest, UnionOverlapping) {
    std::set<int> a = {1, 2, 3}, b = {3, 4, 5}, result;
    SetUnion(a, b, result);
    EXPECT_EQ(result, (std::set<int>{1, 2, 3, 4, 5}));
}

TEST(SetOpsTest, UnionDisjoint) {
    std::set<int> a = {1, 2}, b = {3, 4}, result;
    SetUnion(a, b, result);
    EXPECT_EQ(result.size(), 4u);
}

TEST(SetOpsTest, UnionWithEmpty) {
    std::set<int> a, b = {1, 2}, result;
    SetUnion(a, b, result);
    EXPECT_EQ(result, b);
}

TEST(SetOpsTest, UnionBothEmpty) {
    std::set<int> a, b, result;
    SetUnion(a, b, result);
    EXPECT_TRUE(result.empty());
}

// --- Intersection Tests ---

TEST(SetOpsTest, IntersectionOverlapping) {
    std::set<int> a = {1, 2, 3, 4}, b = {3, 4, 5, 6}, result;
    SetIntersection(a, b, result);
    EXPECT_EQ(result, (std::set<int>{3, 4}));
}

TEST(SetOpsTest, IntersectionDisjoint) {
    std::set<int> a = {1, 2}, b = {3, 4}, result;
    SetIntersection(a, b, result);
    EXPECT_TRUE(result.empty());
}

TEST(SetOpsTest, IntersectionWithEmpty) {
    std::set<int> a = {1, 2}, b, result;
    SetIntersection(a, b, result);
    EXPECT_TRUE(result.empty());
}

// --- Difference Tests ---

TEST(SetOpsTest, DifferenceOverlapping) {
    std::set<int> a = {1, 2, 3, 4}, b = {3, 4, 5}, result;
    SetDifference(a, b, result);
    EXPECT_EQ(result, (std::set<int>{1, 2}));
}

TEST(SetOpsTest, DifferenceNoOverlap) {
    std::set<int> a = {1, 2}, b = {3, 4}, result;
    SetDifference(a, b, result);
    EXPECT_EQ(result, a);
}

TEST(SetOpsTest, DifferenceSubset) {
    std::set<int> a = {1, 2, 3}, b = {1, 2, 3, 4}, result;
    SetDifference(a, b, result);
    EXPECT_TRUE(result.empty());
}

// --- String Type ---

TEST(SetOpsTest, StringUnion) {
    std::set<std::string> a = {"hello", "world"};
    std::set<std::string> b = {"world", "foo"};
    std::set<std::string> result;
    SetUnion(a, b, result);
    EXPECT_EQ(result.size(), 3u);
    EXPECT_TRUE(result.count("hello"));
    EXPECT_TRUE(result.count("foo"));
}
