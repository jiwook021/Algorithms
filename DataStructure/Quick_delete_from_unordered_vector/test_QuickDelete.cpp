#include <gtest/gtest.h>
#include "QuickDelete.hpp"
#include <algorithm>

TEST(QuickDeleteTest, RemoveByIndex) {
    std::vector<int> v{123, 456, 789, 100, 200};
    QuickRemoveAt(v, 2);  // removes 789, replaced by 200
    EXPECT_EQ(v.size(), 4u);
    EXPECT_EQ(std::find(v.begin(), v.end(), 789), v.end());
}

TEST(QuickDeleteTest, RemoveByIterator) {
    std::vector<int> v{10, 20, 30, 40, 50};
    auto It = std::find(v.begin(), v.end(), 30);
    QuickRemoveAt(v, It);
    EXPECT_EQ(v.size(), 4u);
    EXPECT_EQ(std::find(v.begin(), v.end(), 30), v.end());
}

TEST(QuickDeleteTest, RemoveFirstElement) {
    std::vector<int> v{1, 2, 3};
    QuickRemoveAt(v, 0u);
    EXPECT_EQ(v.size(), 2u);
}

TEST(QuickDeleteTest, RemoveLastElement) {
    std::vector<int> v{1, 2, 3};
    QuickRemoveAt(v, 2u);
    EXPECT_EQ(v.size(), 2u);
    // 3 was last, just pop_back
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
}

TEST(QuickDeleteTest, RemoveFromEmptyVector) {
    std::vector<int> v;
    QuickRemoveAt(v, 0u);  // should not crash
    EXPECT_TRUE(v.empty());
}

TEST(QuickDeleteTest, RemoveEndIterator) {
    std::vector<int> v{1, 2, 3};
    QuickRemoveAt(v, v.end());  // no-op
    EXPECT_EQ(v.size(), 3u);
}

TEST(QuickDeleteTest, RemoveAllElements) {
    std::vector<int> v{10, 20, 30};
    QuickRemoveAt(v, 0u);
    QuickRemoveAt(v, 0u);
    QuickRemoveAt(v, 0u);
    EXPECT_TRUE(v.empty());
}

TEST(QuickDeleteTest, StringVector) {
    std::vector<std::string> v{"apple", "banana", "cherry"};
    QuickRemoveAt(v, 1u);
    EXPECT_EQ(v.size(), 2u);
    EXPECT_EQ(std::find(v.begin(), v.end(), "banana"), v.end());
}
