#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <functional>

// BST adapted from Binary_Search_Tree_and_AVL_Tree
struct BtreeNode {
    int data; BtreeNode* left; BtreeNode* right;
    BtreeNode(int d) : data(d), left(nullptr), right(nullptr) {}
};

BtreeNode* BSTInsert(BtreeNode* root, int data) {
    if (!root) return new BtreeNode(data);
    if (data < root->data) root->left = BSTInsert(root->left, data);
    else if (data > root->data) root->right = BSTInsert(root->right, data);
    return root;
}

BtreeNode* BSTSearch(BtreeNode* root, int target) {
    if (!root || root->data == target) return root;
    if (target < root->data) return BSTSearch(root->left, target);
    return BSTSearch(root->right, target);
}

void Inorder(BtreeNode* root, std::vector<int>& result) {
    if (!root) return;
    Inorder(root->left, result);
    result.push_back(root->data);
    Inorder(root->right, result);
}

void FreeTree(BtreeNode* root) {
    if (!root) return;
    FreeTree(root->left); FreeTree(root->right); delete root;
}

TEST(BST, InsertAndSearch) {
    BtreeNode* root = nullptr;
    root = BSTInsert(root, 5);
    root = BSTInsert(root, 3);
    root = BSTInsert(root, 7);
    EXPECT_NE(BSTSearch(root, 5), nullptr);
    EXPECT_NE(BSTSearch(root, 3), nullptr);
    EXPECT_EQ(BSTSearch(root, 10), nullptr);
    FreeTree(root);
}

TEST(BST, InorderSorted) {
    BtreeNode* root = nullptr;
    for (int v : {5, 3, 7, 1, 4}) root = BSTInsert(root, v);
    std::vector<int> result; Inorder(root, result);
    EXPECT_EQ(result, (std::vector<int>{1, 3, 4, 5, 7}));
    FreeTree(root);
}

TEST(BST, SingleNode) {
    BtreeNode* root = BSTInsert(nullptr, 42);
    EXPECT_EQ(root->data, 42);
    EXPECT_EQ(root->left, nullptr);
    FreeTree(root);
}

TEST(BST, DuplicateIgnored) {
    BtreeNode* root = nullptr;
    root = BSTInsert(root, 5);
    root = BSTInsert(root, 5);
    std::vector<int> result; Inorder(root, result);
    EXPECT_EQ(result.size(), 1u);
    FreeTree(root);
}
