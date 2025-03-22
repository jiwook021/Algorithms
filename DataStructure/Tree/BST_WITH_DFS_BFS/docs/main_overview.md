# Code Overview: main.c

This C code implements a **Binary Search Tree (BST)** data structure and provides functionality to insert elements into the tree and print its contents using two different traversal methods: **Depth-First Search (DFS)** and **Breadth-First Search (BFS)**. Let’s break down the purpose, functionality, and structure of the code in detail.

---

### **Purpose of the Code**
The code is designed to:
1. **Create and manage a Binary Search Tree (BST):** A BST is a tree data structure where each node has at most two children, referred to as the left and right child. The key property of a BST is that for any given node:
   - All values in the left subtree are less than the node's value.
   - All values in the right subtree are greater than the node's value.
2. **Insert elements into the BST:** The code allows you to insert integers into the BST while maintaining the BST property.
3. **Traverse and print the BST:** The code provides two methods to traverse and print the tree:
   - **DFS (Depth-First Search):** Specifically, an **in-order traversal** is used, which prints the nodes in ascending order.
   - **BFS (Breadth-First Search):** This prints the nodes level by level, starting from the root.

---

### **Main Functionality**
1. **BST Structure:**
   - The BST is represented using two structs:
     - `Node`: Represents a single node in the tree. It contains:
       - `left`: A pointer to the left child.
       - `right`: A pointer to the right child.
       - `data`: The integer value stored in the node.
     - `Tree`: Represents the entire tree. It contains:
       - `root`: A pointer to the root node of the tree.
2. **Tree Initialization:**
   - The `inittree()` function initializes an empty tree by allocating memory for the `Tree` struct and setting its `root` to `NULL`.
3. **Insertion into the BST:**
   - The `insertBST()` function inserts a new integer into the BST while maintaining the BST property. It:
     - Creates a new node for the integer.
     - Traverses the tree to find the correct position for the new node.
     - Links the new node to its parent.
4. **Tree Traversal and Printing:**
   - **DFS (In-Order Traversal):**
     - The `printTree()` function recursively traverses the tree in an in-order manner (left subtree → root → right subtree) and prints the node values.
   - **BFS (Level-Order Traversal):**
     - The `bfsprint()` function uses a queue to traverse the tree level by level, starting from the root. It prints the nodes in the order they are visited.

---

### **Algorithms Used**
1. **Binary Search Tree Insertion:**
   - The insertion algorithm ensures that the BST property is maintained. It compares the new value with the current node and moves left or right accordingly until it finds the correct position for the new node.
2. **Depth-First Search (DFS):**
   - The in-order traversal algorithm recursively visits the left subtree, then the root, and finally the right subtree. This results in the nodes being printed in ascending order.
3. **Breadth-First Search (BFS):**
   - The BFS algorithm uses a queue to visit nodes level by level. It starts with the root, then visits all nodes at the current level before moving to the next level.

---

### **Overall Structure**
The code is organized into the following components:
1. **Struct Definitions:**
   - `Node` and `Tree` structs define the structure of the BST.
2. **Tree Initialization:**
   - `inittree()` creates and initializes an empty tree.
3. **BST Insertion:**
   - `insertBST()` inserts a new value into the BST.
4. **Tree Traversal:**
   - `printTree()` performs an in-order DFS traversal.
   - `bfsprint()` performs a BFS traversal.
5. **Main Function:**
   - The `main()` function demonstrates the usage of the BST by:
     - Creating a tree.
     - Inserting values into the tree.
     - Printing the tree using BFS.

---

### **How the Parts Work Together**
1. The `main()` function initializes a tree using `inittree()`.
2. It inserts several integers into the tree using `insertBST()`.
3. The tree is then printed using `bfsprint()`, which performs a BFS traversal and prints the nodes level by level.
4. The commented-out code in `main()` shows an alternative example with more insertions and both DFS and BFS traversals.

---

### **Problem Being Solved**
The code solves the problem of **storing and organizing data in a sorted manner** using a Binary Search Tree. It also demonstrates how to traverse and print the tree in two different ways:
1. **DFS (In-Order Traversal):** Useful for printing the tree in sorted order.
2. **BFS (Level-Order Traversal):** Useful for visualizing the tree level by level.

---

### **Approach Taken**
1. **Modular Design:**
   - The code is modular, with separate functions for initialization, insertion, and traversal.
2. **Dynamic Memory Allocation:**
   - The tree and nodes are dynamically allocated using `malloc()`, allowing the tree to grow as needed.
3. **Recursion and Iteration:**
   - DFS traversal uses recursion, while BFS traversal uses iteration with a queue.

---

### **Summary**
This code is a well-structured implementation of a Binary Search Tree. It demonstrates how to:
- Initialize a tree.
- Insert elements while maintaining the BST property.
- Traverse and print the tree using DFS and BFS algorithms.

The code is educational and can be extended or modified for more advanced use cases, such as deleting nodes, balancing the tree, or performing other types of traversals.