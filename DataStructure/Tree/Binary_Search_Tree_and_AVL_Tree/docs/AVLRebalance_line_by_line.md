# Step-by-Step Explanation: AVLRebalance.cpp

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple terms, examples, and diagrams to make everything clear, even for beginners.

---

### **1. Header Files and Includes**
```cpp
#include "AVLRebalance.hpp"
#include "Binary_Search_Tree.hpp"
#include "Binary_tree.h"
#include <iostream>
```

#### **What It Does**
- These lines include necessary header files for the program.
- `AVLRebalance.hpp` likely contains declarations for the functions in this file.
- `Binary_Search_Tree.hpp` and `Binary_tree.h` likely define the `btreeNode` structure and helper functions like `GetLeftTree`, `GetRightTree`, `ChangeLeftTree`, and `ChangeRightTree`.
- `<iostream>` is included for input/output operations (though it’s not used in this code).

#### **Why It’s Used**
- Header files allow us to organize code into reusable modules.
- Including these files ensures the compiler knows about the `btreeNode` structure and helper functions used in this code.

---

### **2. `GetHeight` Function**
```cpp
int GetHeight(btreeNode* Binary_Search_Tree)
{
    int leftH, rightH;
    if (Binary_Search_Tree == NULL)
        return 0;
    leftH = GetHeight(GetLeftTree(Binary_Search_Tree));
    rightH = GetHeight(GetRightTree(Binary_Search_Tree));
    if (leftH > rightH)
        return leftH + 1;
    else
        return rightH + 1;
}
```

#### **What It Does**
- This function calculates the **height** of a binary tree.
- The height of a tree is the number of edges from the root to the deepest leaf.

#### **Step-by-Step Breakdown**
1. **Base Case**:
   - If the tree is empty (`Binary_Search_Tree == NULL`), return `0`. This is because an empty tree has no height.

2. **Recursive Case**:
   - Calculate the height of the **left subtree** by calling `GetHeight(GetLeftTree(Binary_Search_Tree))`.
   - Calculate the height of the **right subtree** by calling `GetHeight(GetRightTree(Binary_Search_Tree))`.

3. **Return the Maximum Height**:
   - Compare the heights of the left and right subtrees.
   - Return the larger height plus `1` (to account for the current node).

#### **Why It’s Used**
- The height of a tree is needed to calculate the **balance factor** (the difference in heights between the left and right subtrees).
- This function is **recursive** because trees are recursive data structures: each subtree is itself a tree.

#### **Example**
Consider this tree:
```
      5
     / \
    3   8
   / \
  2   4
```
- The height of the tree is `2` (edges from `5` to `2` or `5` to `4`).

---

### **3. `getHeightDiff` Function**
```cpp
int getHeightDiff(btreeNode* Binary_Search_Tree)
{
    int left_search_height, right_search_height;

    if (Binary_Search_Tree == NULL)
        return 0;

    left_search_height = GetHeight(GetLeftTree(Binary_Search_Tree));
    right_search_height = GetHeight(GetRightTree(Binary_Search_Tree));
    return left_search_height - right_search_height;
}
```

#### **What It Does**
- This function calculates the **balance factor** of a node.
- The balance factor is the difference in heights between the left and right subtrees.

#### **Step-by-Step Breakdown**
1. **Base Case**:
   - If the tree is empty (`Binary_Search_Tree == NULL`), return `0`. An empty tree is balanced.

2. **Calculate Heights**:
   - Use `GetHeight` to calculate the height of the left subtree (`left_search_height`).
   - Use `GetHeight` to calculate the height of the right subtree (`right_search_height`).

3. **Return the Difference**:
   - Return `left_search_height - right_search_height`.

#### **Why It’s Used**
- The balance factor determines whether a tree is balanced or unbalanced.
- If the balance factor is greater than `1` or less than `-1`, the tree is unbalanced and needs rebalancing.

#### **Example**
For the tree:
```
      5
     / \
    3   8
   / \
  2   4
```
- The height of the left subtree (`3`) is `1`.
- The height of the right subtree (`8`) is `0`.
- The balance factor is `1 - 0 = 1` (balanced).

---

### **4. Rotation Functions**
The code defines four rotation functions: `RotateLL`, `RotateRR`, `RotateLR`, and `RotateRL`. These functions restructure the tree to restore balance.

#### **What They Do**
- **LL Rotation (Left-Left Rotation)**:
  - Used when the left subtree of a node is taller than the right subtree, and the left child's left subtree is also taller.
- **RR Rotation (Right-Right Rotation)**:
  - Used when the right subtree of a node is taller than the left subtree, and the right child's right subtree is also taller.
- **LR Rotation (Left-Right Rotation)**:
  - Used when the left subtree of a node is taller than the right subtree, but the left child's right subtree is taller.
- **RL Rotation (Right-Left Rotation)**:
  - Used when the right subtree of a node is taller than the left subtree, but the right child's left subtree is taller.

#### **Why They’re Used**
- Rotations are the core mechanism for rebalancing an AVL Tree.
- They ensure that the tree remains balanced after insertions or deletions.

---

### **5. `Rebalance` Function**
```cpp
btreeNode* Rebalance(btreeNode** Parent_Root)
{
    int Height_Difference = getHeightDiff(*Parent_Root);

    if (Height_Difference > 1)
    {
        if (getHeightDiff(GetLeftTree((*Parent_Root))) > 0)
            *Parent_Root = RotateLL(*Parent_Root);
        else
            *Parent_Root = RotateLR(*Parent_Root);
    }

    if (Height_Difference < -1)
    {
        if (getHeightDiff(GetRightTree((*Parent_Root))) < 0)
            *Parent_Root = RotateRR(*Parent_Root);
        else
            *Parent_Root = RotateRL(*Parent_Root);
    }
    return *Parent_Root;
}
```

#### **What It Does**
- This function checks if the tree is unbalanced and applies the appropriate rotation(s) to restore balance.

#### **Step-by-Step Breakdown**
1. **Calculate Balance Factor**:
   - Use `getHeightDiff` to calculate the balance factor of the root node.

2. **Left-Heavy Tree**:
   - If the balance factor is greater than `1`, the tree is left-heavy.
   - Check if the left subtree is also left-heavy (`getHeightDiff(GetLeftTree(*Parent_Root)) > 0`).
     - If yes, perform an **LL Rotation**.
     - If no, perform an **LR Rotation**.

3. **Right-Heavy Tree**:
   - If the balance factor is less than `-1`, the tree is right-heavy.
   - Check if the right subtree is also right-heavy (`getHeightDiff(GetRightTree(*Parent_Root)) < 0`).
     - If yes, perform an **RR Rotation**.
     - If no, perform an **RL Rotation**.

4. **Return the New Root**:
   - After rebalancing, return the new root of the tree.

#### **Why It’s Used**
- This function ensures that the tree remains balanced after insertions or deletions.
- It uses the balance factor to determine the type of imbalance and applies the appropriate rotation(s).

---

### **6. Summary**
This code implements AVL Tree rebalancing, ensuring that the tree remains balanced after insertions or deletions. It uses:
- **Height Calculation** to determine the balance factor.
- **Rotations** to restructure the tree and restore balance.
- **Rebalancing Logic** to apply the appropriate rotations based on the balance factor.

By maintaining balance, the tree guarantees **O(log n)** time complexity for search, insert, and delete operations. This is crucial for efficient data storage and retrieval in applications like databases and search engines.

Let me know if you’d like further clarification or examples!