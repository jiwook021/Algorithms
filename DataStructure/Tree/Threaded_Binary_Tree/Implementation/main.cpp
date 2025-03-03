/**
 * @file main.cpp
 * @brief Driver for ThreadedBinaryTree
 */

#include "ThreadedBinaryTree.hpp"
#include <iostream>

int main() {
    ThreadedBinaryTree tree;

    // Insert values
    int values[] = {20, 10, 30, 5, 15, 25, 35};
    for (int v : values) {
        tree.Insert(v);
    }

    std::cout << "Threaded Binary Tree\n";
    std::cout << "Size: " << tree.Size() << "\n";

    std::cout << "In-order traversal (using threads, no recursion): ";
    for (int v : tree.Inorder()) {
        std::cout << v << " ";
    }
    std::cout << "\n";

    std::cout << "Search for 15: " << (tree.Search(15) ? "found" : "not found") << "\n";
    std::cout << "Search for 42: " << (tree.Search(42) ? "found" : "not found") << "\n";

    // Duplicate insert
    bool inserted = tree.Insert(10);
    std::cout << "Insert duplicate 10: " << (inserted ? "inserted" : "ignored") << "\n";

    return 0;
}
