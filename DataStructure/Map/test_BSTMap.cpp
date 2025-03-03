/**
 * @file test_BSTMap.cpp
 * @brief Google Test unit tests for BSTMap (Binary Search Tree Map)
 * @details Educational tests with PrintSection explaining the "why" behind
 *          each test group: insert/find correctness, in-order traversal via
 *          iterator, and erase of a node with two children.
 */

#include "BSTMap.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Educational helper
// ---------------------------------------------------------------------------
namespace {
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}
} // namespace

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------
class BSTMapTest : public ::testing::Test {
protected:
    BSTMap<int, std::string> map_;
    void SetUp() override { map_.clear(); }
};

// ---------------------------------------------------------------------------
// 1. Insert and Find
// ---------------------------------------------------------------------------
TEST_F(BSTMapTest, InsertAndFind) {
    PrintSection("Insert and Find",
                 "Verifies that insert places keys correctly and find retrieves "
                 "them.  Also checks that missing keys return nullopt, ensuring "
                 "the BST search path terminates properly.");

    // Insert three key-value pairs
    EXPECT_TRUE(map_.insert(5, "five"));
    EXPECT_TRUE(map_.insert(2, "two"));
    EXPECT_TRUE(map_.insert(8, "eight"));

    // Successful lookups
    ASSERT_TRUE(map_.find(5).has_value());
    EXPECT_EQ(map_.find(5).value(), "five");

    ASSERT_TRUE(map_.find(2).has_value());
    EXPECT_EQ(map_.find(2).value(), "two");

    ASSERT_TRUE(map_.find(8).has_value());
    EXPECT_EQ(map_.find(8).value(), "eight");

    // Missing key must return nullopt
    EXPECT_FALSE(map_.find(99).has_value());

    // Duplicate insert updates the value, returns false (not a new insertion)
    EXPECT_FALSE(map_.insert(5, "FIVE"));
    EXPECT_EQ(map_.find(5).value(), "FIVE");

    // Size reflects unique keys only
    EXPECT_EQ(map_.size(), 3u);

    std::cout << "  Inserted 3 keys, verified find and duplicate-update.\n";
}

// ---------------------------------------------------------------------------
// 2. In-Order Traversal via Iterator
// ---------------------------------------------------------------------------
TEST_F(BSTMapTest, InOrderTraversalViaIterator) {
    PrintSection("In-Order Traversal via Iterator",
                 "A BST's defining property is that an in-order walk visits keys "
                 "in ascending order.  This test inserts keys out of order and "
                 "confirms the iterator yields them sorted.");

    // Insert keys in non-sorted order
    map_.insert(30, "thirty");
    map_.insert(10, "ten");
    map_.insert(50, "fifty");
    map_.insert(20, "twenty");
    map_.insert(40, "forty");

    // Collect keys via range-for (uses begin()/end())
    std::vector<int> keys;
    std::vector<std::string> values;
    for (const auto& pair : map_) {
        keys.push_back(pair.first);
        values.push_back(pair.second);
    }

    // Keys must be in ascending order
    const std::vector<int> kExpectedKeys = {10, 20, 30, 40, 50};
    EXPECT_EQ(keys, kExpectedKeys);

    // Values follow the same sorted-key order
    const std::vector<std::string> kExpectedValues = {
        "ten", "twenty", "thirty", "forty", "fifty"
    };
    EXPECT_EQ(values, kExpectedValues);

    // Prefix ++ must advance correctly through every node
    auto it = map_.begin();
    for (size_t i = 0; i < kExpectedKeys.size(); ++i, ++it) {
        EXPECT_EQ(it->first, kExpectedKeys[i]);
    }
    EXPECT_EQ(it, map_.end());

    std::cout << "  Iterator visited " << keys.size()
              << " nodes in sorted order.\n";
}

// ---------------------------------------------------------------------------
// 3. Erase Node with Two Children
// ---------------------------------------------------------------------------
TEST_F(BSTMapTest, EraseNodeWithTwoChildren) {
    PrintSection("Erase Node with Two Children",
                 "Removing an internal node that has both left and right children "
                 "is the hardest BST deletion case.  The algorithm replaces the "
                 "node's data with its in-order successor, then recursively "
                 "removes the successor.  This test verifies structural integrity "
                 "and correct size after such a removal.");

    //        5
    //       / \        Build a tree where key 5 has two children
    //      3   8
    //     / \    \     (backslash is the right-child edge)
    //    1   4    9
    map_.insert(5, "five");
    map_.insert(3, "three");
    map_.insert(8, "eight");
    map_.insert(1, "one");
    map_.insert(4, "four");
    map_.insert(9, "nine");

    ASSERT_EQ(map_.size(), 6u);

    // Remove key 5 -- it has two children (3 and 8)
    EXPECT_TRUE(map_.remove(5));
    EXPECT_EQ(map_.size(), 5u);
    EXPECT_FALSE(map_.Contains(5));

    // All other keys must still be reachable
    EXPECT_TRUE(map_.Contains(1));
    EXPECT_TRUE(map_.Contains(3));
    EXPECT_TRUE(map_.Contains(4));
    EXPECT_TRUE(map_.Contains(8));
    EXPECT_TRUE(map_.Contains(9));

    // In-order traversal must still be sorted after the deletion
    std::vector<int> keys;
    for (const auto& pair : map_) {
        keys.push_back(pair.first);
    }
    const std::vector<int> kExpectedAfterErase = {1, 3, 4, 8, 9};
    EXPECT_EQ(keys, kExpectedAfterErase);

    // Also remove key 3 (it has two children: 1 and 4)
    EXPECT_TRUE(map_.remove(3));
    EXPECT_EQ(map_.size(), 4u);
    EXPECT_FALSE(map_.Contains(3));
    EXPECT_TRUE(map_.Contains(1));
    EXPECT_TRUE(map_.Contains(4));

    std::cout << "  Removed two-child nodes 5 and 3; tree remains valid.\n";
}
