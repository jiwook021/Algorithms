/**
 * @file main.cpp
 * @brief Driver for AvlTree (AVL-balanced Binary Search Tree)
 */

#include "AvlTree.hpp"
#include <iostream>

int main() {
    AvlTree<int> tree;

    // Insert values 1 through 7
    for (int i = 1; i <= 7; ++i) {
        tree.Insert(i);
    }

    std::cout << "Tree size: " << tree.Size() << "\n";
    std::cout << "Tree height: " << tree.TreeHeight() << "\n";

    std::cout << "\nIn-order traversal: ";
    for (int v : tree.Inorder()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "Pre-order traversal: ";
    for (int v : tree.Preorder()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "Post-order traversal: ";
    for (int v : tree.Postorder()) std::cout << v << " ";
    std::cout << "\n";

    std::cout << "\nSearch for 6: " << (tree.Search(6) ? "found" : "not found") << "\n";
    std::cout << "Search for 21: " << (tree.Search(21) ? "found" : "not found") << "\n";

    tree.Remove(4);
    std::cout << "\nAfter removing 4, in-order: ";
    for (int v : tree.Inorder()) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}
