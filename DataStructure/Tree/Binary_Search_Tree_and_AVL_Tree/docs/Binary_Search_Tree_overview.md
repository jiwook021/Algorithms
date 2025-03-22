# Code Overview: Binary_Search_Tree.cpp

This C++ code implements a **Binary Search Tree (BST)** with additional **AVL Tree** balancing functionality. Let's break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The code provides a **dynamic data structure** for storing and managing a collection of data (integers in this case) in a way that allows for efficient searching, insertion, and deletion. The BST is a tree-based data structure where each node has at most two children, and the tree is organized such that:
- All nodes in the **left subtree** of a node contain values **less than** the node's value.
- All nodes in the **right subtree** of a node contain values **greater than** the node's value.

The code also includes **AVL Tree rebalancing**, which ensures that the tree remains balanced after insertions and deletions. A balanced tree guarantees that operations like search, insertion, and deletion take **O(log n)** time in the worst case, where `n` is the number of nodes in the tree.

---

### **Main Functionality**
The code provides the following core functionalities:
1. **Initialization**: Create and initialize an empty BST.
2. **Insertion**: Insert a new node with a given value into the BST while maintaining the BST property and ensuring the tree remains balanced (AVL).
3. **Search**: Search for a node with a specific value in the BST.
4. **Deletion**: Remove a node with a specific value from the BST while maintaining the BST property and ensuring the tree remains balanced (AVL).

---

### **Algorithms Used**
1. **Binary Search Tree Operations**:
   - **Insertion**: Recursively finds the correct position for a new node based on the BST property.
   - **Search**: Iteratively traverses the tree to find a node with the target value.
   - **Deletion**: Handles three cases:
     - Node has no children (leaf node).
     - Node has one child.
     - Node has two children (replaces the node with its in-order successor).

2. **AVL Tree Rebalancing**:
   - After insertion or deletion, the tree is checked for balance using the `Rebalance` function (likely implemented in `AVLRebalance.hpp`).
   - AVL trees maintain a balance factor (height difference between left and right subtrees) of -1, 0, or 1 for every node. If the balance factor is violated, rotations (single or double) are performed to restore balance.

---

### **Overall Structure**
The code is organized into several functions, each responsible for a specific operation:
1. **BSTMakeAndInit**: Initializes the BST by setting the root node to `nullptr`.
2. **BSTGetNodeData**: Retrieves the data stored in a given node.
3. **BSTInsert**: Inserts a new node into the BST and rebalances the tree if necessary.
4. **BSTSearch**: Searches for a node with a specific value in the BST.
5. **BSTRemove**: Removes a node with a specific value from the BST and rebalances the tree if necessary.

---

### **How the Parts Work Together**
1. **Initialization**:
   - The `BSTMakeAndInit` function sets up an empty BST by initializing the root node to `nullptr`.

2. **Insertion**:
   - The `BSTInsert` function recursively finds the correct position for a new node based on the BST property.
   - After insertion, the `Rebalance` function is called to ensure the tree remains balanced (AVL).

3. **Search**:
   - The `BSTSearch` function iteratively traverses the tree to find a node with the target value. It uses the BST property to decide whether to search the left or right subtree.

4. **Deletion**:
   - The `BSTRemove` function handles three cases for node removal:
     - If the node has no children, it is simply removed.
     - If the node has one child, the child replaces the node.
     - If the node has two children, it is replaced by its in-order successor.
   - After deletion, the `Rebalance` function is called to ensure the tree remains balanced (AVL).

---

### **Problem Being Solved**
The code solves the problem of efficiently managing a dynamic set of data (integers) with the following requirements:
1. **Fast Search**: Quickly find whether a value exists in the set.
2. **Dynamic Updates**: Efficiently insert and delete values while maintaining the BST property.
3. **Balanced Tree**: Ensure the tree remains balanced to guarantee O(log n) time complexity for all operations.

---

### **Approach Taken**
1. **BST Operations**:
   - The code uses standard BST algorithms for insertion, search, and deletion.
   - For deletion, it handles all three cases (no children, one child, two children) to maintain the BST property.

2. **AVL Balancing**:
   - After every insertion or deletion, the `Rebalance` function is called to check and restore the tree's balance.
   - This ensures that the tree does not degenerate into a linked list (which would result in O(n) time complexity for operations).

3. **Memory Management**:
   - The code uses `malloc` for dynamic memory allocation and `free` for deallocation.
   - This is a C-style approach; in modern C++, `new` and `delete` or smart pointers would be preferred.

---

### **Key Observations**
1. **AVL Integration**:
   - The `Rebalance` function (likely defined in `AVLRebalance.hpp`) is called after every insertion and deletion to maintain the AVL property.

2. **C-Style Code**:
   - The code uses C-style memory management (`malloc` and `free`) and pointer manipulation, which is less common in modern C++.

3. **Error Handling**:
   - The code does not explicitly handle memory allocation failures (e.g., if `malloc` returns `nullptr`).

4. **Tree Traversal**:
   - The search and deletion functions use iterative traversal, which is efficient and avoids the overhead of recursive function calls.

---

### **Summary**
This code implements a **Binary Search Tree** with **AVL balancing** to ensure efficient operations. It provides functions for initialization, insertion, search, and deletion, all while maintaining the BST property and ensuring the tree remains balanced. The integration of AVL balancing guarantees optimal performance for large datasets. However, the code could be modernized to use C++ features like smart pointers and exception handling.

Let me know if you'd like a line-by-line explanation or suggestions for improvements!