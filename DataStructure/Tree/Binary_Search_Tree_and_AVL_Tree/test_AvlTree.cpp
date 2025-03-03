/**
 * @file test_AvlTree.cpp
 * @brief Unit tests for AvlTree (AVL-balanced Binary Search Tree)
 */

#include "AvlTree.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

TEST(AvlTreeTest, EmptyTree) {
    AvlTree<int> tree;
    EXPECT_TRUE(tree.Empty());
    EXPECT_EQ(tree.Size(), 0u);
    EXPECT_EQ(tree.TreeHeight(), 0);
}

TEST(AvlTreeTest, InsertAndSearch) {
    AvlTree<int> tree;
    tree.Insert(5);
    tree.Insert(3);
    tree.Insert(7);

    EXPECT_TRUE(tree.Search(5));
    EXPECT_TRUE(tree.Search(3));
    EXPECT_TRUE(tree.Search(7));
    EXPECT_FALSE(tree.Search(10));
    EXPECT_EQ(tree.Size(), 3u);
}

TEST(AvlTreeTest, InorderSorted) {
    AvlTree<int> tree;
    for (int v : {5, 3, 7, 1, 4}) {
        tree.Insert(v);
    }
    auto result = tree.Inorder();
    EXPECT_EQ(result, (std::vector<int>{1, 3, 4, 5, 7}));
}

TEST(AvlTreeTest, DuplicateIgnored) {
    AvlTree<int> tree;
    tree.Insert(5);
    tree.Insert(5);
    EXPECT_EQ(tree.Size(), 1u);
}

TEST(AvlTreeTest, BalancedAfterSequentialInsert) {
    AvlTree<int> tree;
    // Inserting 1..7 sequentially would create an unbalanced BST,
    // but AVL rotations should keep height at O(log n)
    for (int i = 1; i <= 7; ++i) {
        tree.Insert(i);
    }
    EXPECT_EQ(tree.Size(), 7u);
    EXPECT_LE(tree.TreeHeight(), 4);  // log2(7) ~ 2.8, AVL height <= 1.44*log2(n+2)

    auto sorted = tree.Inorder();
    EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));
}

TEST(AvlTreeTest, RemoveLeaf) {
    AvlTree<int> tree{1, 2, 3, 4, 5};
    EXPECT_TRUE(tree.Remove(1));
    EXPECT_FALSE(tree.Search(1));
    EXPECT_EQ(tree.Size(), 4u);
}

TEST(AvlTreeTest, RemoveNodeWithOneChild) {
    AvlTree<int> tree{3, 1, 5, 4};
    EXPECT_TRUE(tree.Remove(5));
    EXPECT_FALSE(tree.Search(5));
    EXPECT_TRUE(tree.Search(4));
}

TEST(AvlTreeTest, RemoveNodeWithTwoChildren) {
    AvlTree<int> tree{3, 1, 5, 4, 6};
    EXPECT_TRUE(tree.Remove(5));
    EXPECT_FALSE(tree.Search(5));

    auto sorted = tree.Inorder();
    EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));
    EXPECT_EQ(sorted.size(), 4u);
}

TEST(AvlTreeTest, RemoveRoot) {
    AvlTree<int> tree{5, 3, 7};
    EXPECT_TRUE(tree.Remove(5));
    EXPECT_FALSE(tree.Search(5));
    EXPECT_EQ(tree.Size(), 2u);
}

TEST(AvlTreeTest, RemoveNonexistent) {
    AvlTree<int> tree{1, 2, 3};
    EXPECT_FALSE(tree.Remove(99));
    EXPECT_EQ(tree.Size(), 3u);
}

TEST(AvlTreeTest, Clear) {
    AvlTree<int> tree{1, 2, 3, 4, 5};
    tree.Clear();
    EXPECT_TRUE(tree.Empty());
    EXPECT_EQ(tree.Size(), 0u);

    // Can insert again after clear
    tree.Insert(10);
    EXPECT_TRUE(tree.Search(10));
}

TEST(AvlTreeTest, PreorderAndPostorder) {
    AvlTree<int> tree{2, 1, 3};
    auto pre = tree.Preorder();
    auto post = tree.Postorder();

    EXPECT_EQ(pre.size(), 3u);
    EXPECT_EQ(post.size(), 3u);
    // Pre-order: root first
    EXPECT_EQ(pre[0], 2);
    // Post-order: root last
    EXPECT_EQ(post[2], 2);
}

TEST(AvlTreeTest, CopyConstructor) {
    AvlTree<int> tree{1, 2, 3, 4, 5};
    AvlTree<int> copy(tree);

    EXPECT_EQ(copy.Size(), 5u);
    EXPECT_EQ(copy.Inorder(), tree.Inorder());

    // Modifying copy does not affect original
    copy.Remove(3);
    EXPECT_TRUE(tree.Search(3));
    EXPECT_FALSE(copy.Search(3));
}

TEST(AvlTreeTest, MoveConstructor) {
    AvlTree<int> tree{1, 2, 3};
    AvlTree<int> moved(std::move(tree));

    EXPECT_EQ(moved.Size(), 3u);
    EXPECT_TRUE(moved.Search(2));
    EXPECT_TRUE(tree.Empty());
}

TEST(AvlTreeTest, InitializerList) {
    AvlTree<int> tree{10, 20, 30, 40, 50};
    EXPECT_EQ(tree.Size(), 5u);
    auto sorted = tree.Inorder();
    EXPECT_EQ(sorted, (std::vector<int>{10, 20, 30, 40, 50}));
}

TEST(AvlTreeTest, LargeSequentialInsert) {
    AvlTree<int> tree;
    for (int i = 0; i < 1000; ++i) {
        tree.Insert(i);
    }
    EXPECT_EQ(tree.Size(), 1000u);
    // AVL height for 1000 nodes should be <= ~15
    EXPECT_LE(tree.TreeHeight(), 15);

    auto sorted = tree.Inorder();
    EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));
}

TEST(AvlTreeTest, SingleNode) {
    AvlTree<int> tree;
    tree.Insert(42);
    EXPECT_EQ(tree.Size(), 1u);
    EXPECT_EQ(tree.TreeHeight(), 1);
    EXPECT_TRUE(tree.Search(42));

    tree.Remove(42);
    EXPECT_TRUE(tree.Empty());
}
