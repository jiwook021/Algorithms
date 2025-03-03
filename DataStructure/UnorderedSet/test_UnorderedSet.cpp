/**
 * @file test_UnorderedSet.cpp
 * @brief Unit tests for SimpleUnorderedSet and List
 */

#include "UnorderedSet.hpp"
#include <gtest/gtest.h>

// --- List Tests ---

class ListTest : public ::testing::Test {
protected:
    List<int> list;
    void SetUp() override {
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);
    }
};

TEST_F(ListTest, InitialSize) {
    EXPECT_EQ(list.size(), 3u);
    EXPECT_FALSE(list.empty());
}

TEST_F(ListTest, PushFront) {
    list.push_front(5);
    EXPECT_EQ(list.size(), 4u);
    EXPECT_EQ(*list.begin(), 5);
}

TEST_F(ListTest, PushBack) {
    list.push_back(40);
    EXPECT_EQ(list.size(), 4u);
}

TEST_F(ListTest, PopFront) {
    list.pop_front();
    EXPECT_EQ(list.size(), 2u);
    EXPECT_EQ(*list.begin(), 20);
}

TEST_F(ListTest, Remove) {
    EXPECT_TRUE(list.remove(20));
    EXPECT_EQ(list.size(), 2u);
    EXPECT_FALSE(list.Contains(20));
    EXPECT_FALSE(list.remove(50));
}

TEST_F(ListTest, Contains) {
    EXPECT_TRUE(list.Contains(10));
    EXPECT_TRUE(list.Contains(30));
    EXPECT_FALSE(list.Contains(40));
}

TEST_F(ListTest, Clear) {
    list.clear();
    EXPECT_TRUE(list.empty());
    EXPECT_EQ(list.size(), 0u);
}

TEST_F(ListTest, InitializerList) {
    List<int> newList = {1, 2, 3, 4, 5};
    EXPECT_EQ(newList.size(), 5u);
    EXPECT_TRUE(newList.Contains(1));
    EXPECT_TRUE(newList.Contains(5));
}

TEST_F(ListTest, CopyConstructor) {
    List<int> copy(list);
    EXPECT_EQ(copy.size(), 3u);
    EXPECT_TRUE(copy.Contains(10));
    EXPECT_TRUE(copy.Contains(30));
}

TEST_F(ListTest, MoveConstructor) {
    List<int> moved(std::move(list));
    EXPECT_EQ(moved.size(), 3u);
    EXPECT_TRUE(moved.Contains(10));
    EXPECT_TRUE(list.empty());
}

// --- SimpleUnorderedSet Tests ---

class SimpleUnorderedSetTest : public ::testing::Test {
protected:
    SimpleUnorderedSet<int> set;
    void SetUp() override {
        set.insert(10);
        set.insert(20);
        set.insert(30);
    }
};

TEST_F(SimpleUnorderedSetTest, InitialSize) {
    EXPECT_EQ(set.size(), 3u);
    EXPECT_FALSE(set.empty());
}

TEST_F(SimpleUnorderedSetTest, Insert) {
    auto [it, inserted] = set.insert(40);
    EXPECT_TRUE(inserted);
    EXPECT_EQ(set.size(), 4u);

    auto [it2, inserted2] = set.insert(10);
    EXPECT_FALSE(inserted2);
    EXPECT_EQ(set.size(), 4u);
}

TEST_F(SimpleUnorderedSetTest, Erase) {
    EXPECT_EQ(set.erase(20), 1u);
    EXPECT_EQ(set.size(), 2u);
    EXPECT_FALSE(set.Contains(20));
    EXPECT_EQ(set.erase(50), 0u);
}

TEST_F(SimpleUnorderedSetTest, Contains) {
    EXPECT_TRUE(set.Contains(10));
    EXPECT_TRUE(set.Contains(30));
    EXPECT_FALSE(set.Contains(40));
}

TEST_F(SimpleUnorderedSetTest, Find) {
    auto it = set.find(20);
    EXPECT_NE(it, set.end());
    EXPECT_EQ(*it, 20);

    auto it2 = set.find(50);
    EXPECT_EQ(it2, set.end());
}

TEST_F(SimpleUnorderedSetTest, Clear) {
    set.clear();
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0u);
}

TEST_F(SimpleUnorderedSetTest, LoadFactor) {
    EXPECT_LE(set.LoadFactor(), set.MaxLoadFactor());
    for (int i = 100; i < 150; ++i) set.insert(i);
    EXPECT_LE(set.LoadFactor(), set.MaxLoadFactor());
}

TEST_F(SimpleUnorderedSetTest, EdgeCases) {
    SimpleUnorderedSet<int> emptySet;
    EXPECT_TRUE(emptySet.empty());
    EXPECT_EQ(emptySet.find(10), emptySet.end());

    set.clear();
    set.insert(100);
    EXPECT_EQ(set.size(), 1u);
    EXPECT_TRUE(set.Contains(100));
}
