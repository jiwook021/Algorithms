/**
 * @file test_KeepVectorSorted.cpp
 * @brief Unit tests for KeepVectorSorted
 */

#include "KeepVectorSorted.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <deque>
#include <string>

using namespace KeepVectorSorted;

TEST(KeepVectorSorted, InsertIntoEmpty) {
    std::vector<int> v;
    InsertSorted(v, 5);
    EXPECT_EQ(v, (std::vector<int>{5}));
}

TEST(KeepVectorSorted, MaintainsSortOrder) {
    std::vector<int> v = {1, 3, 5};
    InsertSorted(v, 2);
    EXPECT_EQ(v, (std::vector<int>{1, 2, 3, 5}));
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(KeepVectorSorted, InsertAtEnd) {
    std::vector<int> v = {1, 2, 3};
    InsertSorted(v, 10);
    EXPECT_EQ(v.back(), 10);
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(KeepVectorSorted, InsertAtBegin) {
    std::vector<int> v = {5, 10, 15};
    InsertSorted(v, 1);
    EXPECT_EQ(v.front(), 1);
}

TEST(KeepVectorSorted, Strings) {
    std::vector<std::string> v = {"apple", "cherry"};
    InsertSorted(v, std::string("banana"));
    EXPECT_EQ(v, (std::vector<std::string>{"apple", "banana", "cherry"}));
}

TEST(KeepVectorSorted, Deque) {
    std::deque<int> d = {2, 4, 6};
    InsertSorted(d, 3);
    EXPECT_TRUE(std::is_sorted(d.begin(), d.end()));
    EXPECT_EQ(d[1], 3);
}

TEST(KeepVectorSorted, Duplicates) {
    std::vector<int> v = {1, 3, 5};
    InsertSorted(v, 3);
    EXPECT_EQ(v.size(), 4u);
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}
