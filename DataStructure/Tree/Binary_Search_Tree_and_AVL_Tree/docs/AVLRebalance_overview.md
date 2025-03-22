# Code Overview: AVLRebalance.cpp

This C++ code is an implementation of **AVL Tree Rebalancing**, which is a crucial operation in maintaining the balance of an **AVL Tree** (Adelson-Velsky and Landis Tree). An AVL Tree is a self-balancing binary search tree (BST) where the difference in heights between the left and right subtrees of any node (called the **balance factor**) is at most 1. If this balance factor exceeds 1, the tree becomes unbalanced, and rebalancing is required to maintain its efficiency.

### **Purpose of the Code**
The purpose of this code is to **rebalance an AVL Tree** whenever it becomes unbalanced due to insertions or deletions. The code ensures that the tree remains balanced by performing **rotations** on nodes where the balance factor exceeds the allowed limit. This guarantees that the tree maintains its **O(log n)** time complexity for search, insert, and delete operations.

---

### **Main Functionality**
The code provides the following key functionalities:
1. **Height Calculation**: Determines the height of a subtree.
2. **Balance Factor Calculation**: Computes the difference in heights between the left and right subtrees of a node.
3. **Rotations**: Performs rotations (LL, RR, LR, RL) to rebalance the tree when the balance factor exceeds the allowed limit.
4. **Rebalancing**: Uses the above functions to rebalance the tree by applying the appropriate rotations.

---

### **Algorithms Used**
1. **Height Calculation**:
   - The height of a tree is the number of edges from the root to the deepest leaf.
   - The `GetHeight` function recursively calculates the height of the left and right subtrees and returns the maximum height plus 1.

2. **Balance Factor Calculation**:
   - The balance factor of a node is the difference between the heights of its left and right subtrees.
   - The `getHeightDiff` function calculates this difference.

3. **Rotations**:
   - **LL Rotation (Left-Left Rotation)**: Used when the left subtree of a node is taller than the right subtree, and the left child's left subtree is also taller.
   - **RR Rotation (Right-Right Rotation)**: Used when the right subtree of a node is taller than the left subtree, and the right child's right subtree is also taller.
   - **LR Rotation (Left-Right Rotation)**: Used when the left subtree of a node is taller than the right subtree, but the left child's right subtree is taller.
   - **RL Rotation (Right-Left Rotation)**: Used when the right subtree of a node is taller than the left subtree, but the right child's left subtree is taller.

4. **Rebalancing**:
   - The `Rebalance` function checks the balance factor of the root node and applies the appropriate rotation(s) to restore balance.

---

### **Overall Structure**
The code is structured into several functions, each responsible for a specific task:
1. **`GetHeight`**: Calculates the height of a tree.
2. **`getHeightDiff`**: Computes the balance factor of a node.
3. **`RotateLL`, `RotateRR`, `RotateLR`, `RotateRL`**: Perform the necessary rotations to rebalance the tree.
4. **`Rebalance`**: Orchestrates the rebalancing process by checking the balance factor and applying the appropriate rotation.

---

### **How the Code Works Together**
1. **Height and Balance Factor**:
   - The `GetHeight` function is used by `getHeightDiff` to determine the heights of the left and right subtrees.
   - The `getHeightDiff` function calculates the balance factor, which is used to determine if the tree is unbalanced.

2. **Rotations**:
   - The rotation functions (`RotateLL`, `RotateRR`, `RotateLR`, `RotateRL`) restructure the tree to restore balance.
   - These functions are called by the `Rebalance` function based on the balance factor.

3. **Rebalancing**:
   - The `Rebalance` function checks the balance factor of the root node.
   - If the tree is unbalanced, it determines the type of imbalance (left-heavy or right-heavy) and applies the appropriate rotation(s).

---

### **Problem Being Solved**
The problem being solved is **maintaining the balance of an AVL Tree** after insertions or deletions. Without rebalancing, the tree could degenerate into a linked list, resulting in **O(n)** time complexity for operations. By ensuring the tree remains balanced, the code guarantees **O(log n)** time complexity for all operations.

---

### **Approach Taken**
The approach taken is to:
1. Calculate the height of each subtree.
2. Compute the balance factor for each node.
3. If the balance factor exceeds the allowed limit, perform the appropriate rotation(s) to restore balance.
4. Repeat this process recursively for all nodes in the tree.

---

### **How the Parts Work Together**
- The `GetHeight` and `getHeightDiff` functions provide the necessary information about the tree's structure.
- The rotation functions restructure the tree to restore balance.
- The `Rebalance` function ties everything together by checking the balance factor and applying the appropriate rotation(s).

This modular approach ensures that the code is clean, maintainable, and easy to understand. Each function has a single responsibility, and they work together to achieve the overall goal of maintaining a balanced AVL Tree.