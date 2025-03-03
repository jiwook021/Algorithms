/**
 * @file test_ThreadedBinaryTree.cpp
 * @brief Unit tests for ThreadedBinaryTree
 */

#include "ThreadedBinaryTree.hpp"
#include <algorithm>
#include <gtest/gtest.h>

TEST(ThreadedBinaryTreeTest, EmptyTree) {
    ThreadedBinaryTree tree;
    EXPECT_TRUE(tree.Empty());
    EXPECT_EQ(tree.Size(), 0u);
    EXPECT_TRUE(tree.Inorder().empty());
}

TEST(ThreadedBinaryTreeTest, InsertAndSearch) {
    ThreadedBinaryTree tree;
    EXPECT_TRUE(tree.Insert(10));
    EXPECT_TRUE(tree.Insert(5));
    EXPECT_TRUE(tree.Insert(15));

    EXPECT_TRUE(tree.Search(10));
    EXPECT_TRUE(tree.Search(5));
    EXPECT_TRUE(tree.Search(15));
    EXPECT_FALSE(tree.Search(20));
    EXPECT_EQ(tree.Size(), 3u);
}

TEST(ThreadedBinaryTreeTest, DuplicateIgnored) {
    ThreadedBinaryTree tree;
    EXPECT_TRUE(tree.Insert(10));
    EXPECT_FALSE(tree.Insert(10));
    EXPECT_EQ(tree.Size(), 1u);
}

TEST(ThreadedBinaryTreeTest, InorderSorted) {
    ThreadedBinaryTree tree;
    for (int v : {20, 10, 30, 5, 15, 25, 35}) {
        tree.Insert(v);
    }

    auto result = tree.Inorder();
    EXPECT_EQ(result, (std::vector<int>{5, 10, 15, 20, 25, 30, 35}));
}

TEST(ThreadedBinaryTreeTest, SequentialInsert) {
    ThreadedBinaryTree tree;
    for (int i = 1; i <= 10; ++i) {
        tree.Insert(i);
    }

    auto result = tree.Inorder();
    EXPECT_EQ(result.size(), 10u);
    EXPECT_TRUE(std::is_sorted(result.begin(), result.end()));
}

TEST(ThreadedBinaryTreeTest, ReverseInsert) {
    ThreadedBinaryTree tree;
    for (int i = 10; i >= 1; --i) {
        tree.Insert(i);
    }

    auto result = tree.Inorder();
    EXPECT_EQ(result, (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}

TEST(ThreadedBinaryTreeTest, SingleElement) {
    ThreadedBinaryTree tree;
    tree.Insert(42);

    EXPECT_EQ(tree.Size(), 1u);
    EXPECT_TRUE(tree.Search(42));
    auto result = tree.Inorder();
    EXPECT_EQ(result, std::vector<int>{42});
}

TEST(ThreadedBinaryTreeTest, Clear) {
    ThreadedBinaryTree tree;
    for (int v : {5, 3, 7, 1, 4}) {
        tree.Insert(v);
    }

    tree.Clear();
    EXPECT_TRUE(tree.Empty());
    EXPECT_EQ(tree.Size(), 0u);
    EXPECT_FALSE(tree.Search(5));

    // Can insert again after clear
    tree.Insert(100);
    EXPECT_TRUE(tree.Search(100));
}

TEST(ThreadedBinaryTreeTest, MoveConstructor) {
    ThreadedBinaryTree tree;
    tree.Insert(10);
    tree.Insert(5);
    tree.Insert(15);

    ThreadedBinaryTree moved(std::move(tree));
    EXPECT_EQ(moved.Size(), 3u);
    EXPECT_TRUE(moved.Search(10));
    EXPECT_TRUE(tree.Empty());
}

TEST(ThreadedBinaryTreeTest, LargeTree) {
    ThreadedBinaryTree tree;
    for (int i = 0; i < 100; ++i) {
        tree.Insert(i);
    }
    EXPECT_EQ(tree.Size(), 100u);

    auto result = tree.Inorder();
    EXPECT_TRUE(std::is_sorted(result.begin(), result.end()));
    EXPECT_EQ(result.size(), 100u);
}
