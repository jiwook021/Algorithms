# Step-by-Step Explanation: Binary_Search_Tree.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the code’s design.

---

### **1. Includes and Dependencies**
```cpp
#include "Binary_Search_Tree.hpp"
#include "AVLRebalance.hpp"
#include <cstdlib>
```

#### **What It Does**
- These lines include necessary header files for the program to work:
  - `Binary_Search_Tree.hpp`: Likely contains the definition of the `btreeNode` structure and helper functions like `GetData`, `GetLeftTree`, `GetRightTree`, etc.
  - `AVLRebalance.hpp`: Contains the `Rebalance` function, which ensures the tree remains balanced after insertions and deletions.
  - `<cstdlib>`: Provides functions like `malloc` and `free` for dynamic memory management.

#### **Why It’s Used**
- Including these files allows the program to use the `btreeNode` structure and helper functions defined elsewhere, keeping the code modular and organized.

---

### **2. Initialization Function**
```cpp
void BSTMakeAndInit(btreeNode** Parent_Root)
{
    *Parent_Root = nullptr; 
}
```

#### **What It Does**
- This function initializes a Binary Search Tree by setting the root node (`Parent_Root`) to `nullptr`, meaning the tree is empty.

#### **Breakdown**
- `btreeNode** Parent_Root`: A pointer to a pointer to a `btreeNode`. This allows the function to modify the root node of the tree.
- `*Parent_Root = nullptr`: Sets the root node to `nullptr`, indicating an empty tree.

#### **Why It’s Used**
- Initialization is necessary to start with a clean, empty tree. Without this, the root pointer could contain garbage values, leading to undefined behavior.

---

### **3. Node Data Retrieval Function**
```cpp
const int BSTGetNodeData(btreeNode* Binary_Search_tree_Node)
{
    return GetData(Binary_Search_tree_Node);
}
```

#### **What It Does**
- This function retrieves the data stored in a given node.

#### **Breakdown**
- `btreeNode* Binary_Search_tree_Node`: A pointer to a node in the BST.
- `GetData(Binary_Search_tree_Node)`: Calls a helper function (likely defined in `Binary_Search_Tree.hpp`) to retrieve the data stored in the node.

#### **Why It’s Used**
- Encapsulates the logic for retrieving node data, making the code cleaner and easier to maintain.

---

### **4. Insertion Function**
```cpp
btreeNode* BSTInsert(btreeNode** Parent_Root, int data)
{
    if (*Parent_Root == NULL)
    {
        *Parent_Root = (btreeNode*) malloc (sizeof(btreeNode));
        initbtreeNode(data,*Parent_Root);
    }
    else if (data < GetData(*Parent_Root))
    {
        BSTInsert(&((*Parent_Root)->left), data);
        *Parent_Root = Rebalance(Parent_Root);
    }
    else if (data > GetData(*Parent_Root))
    {
        BSTInsert(&((*Parent_Root)->right), data);
        *Parent_Root = Rebalance(Parent_Root);
    }
    else
    {
        return NULL;
    }
    return *Parent_Root;
}
```

#### **What It Does**
- Inserts a new node with the given `data` into the BST. If the data already exists, it does nothing. After insertion, it rebalances the tree to maintain the AVL property.

#### **Breakdown**
1. **Base Case**:
   - If the tree is empty (`*Parent_Root == NULL`), allocate memory for a new node using `malloc` and initialize it with `initbtreeNode`.

2. **Recursive Cases**:
   - If `data` is less than the current node’s data, recursively insert into the left subtree.
   - If `data` is greater than the current node’s data, recursively insert into the right subtree.

3. **Rebalancing**:
   - After insertion, call `Rebalance` to ensure the tree remains balanced.

4. **Duplicate Data**:
   - If `data` is equal to the current node’s data, return `NULL` (no duplicates allowed).

#### **Why It’s Used**
- Recursive insertion ensures the new node is placed in the correct position based on the BST property. Rebalancing maintains the AVL property, ensuring efficient operations.

#### **Example**
Suppose we insert `[5, 3, 7]` into an empty tree:
1. Insert `5`: Root node is `5`.
2. Insert `3`: `3` is less than `5`, so it goes to the left of `5`.
3. Insert `7`: `7` is greater than `5`, so it goes to the right of `5`.

The tree looks like this:
```
    5
   / \
  3   7
```

---

### **5. Search Function**
```cpp
btreeNode* BSTSearch(btreeNode* Binary_Search_tree_Node, int target)
{
    btreeNode* Child_Node = Binary_Search_tree_Node; 
    int cd;
    while (Child_Node != nullptr)
    {
        cd = GetData(Child_Node);
        if (target == cd)
            return Child_Node;
        else if (target < cd)
            Child_Node = GetLeftTree(Child_Node);
        else
            Child_Node = GetRightTree(Child_Node);
    }
    return nullptr; 
}
```

#### **What It Does**
- Searches for a node with the given `target` value in the BST.

#### **Breakdown**
1. **Initialization**:
   - Start at the root node (`Binary_Search_tree_Node`).

2. **Traversal**:
   - Use a `while` loop to traverse the tree:
     - If `target` equals the current node’s data, return the node.
     - If `target` is less, move to the left child.
     - If `target` is greater, move to the right child.

3. **Termination**:
   - If the loop ends without finding the target, return `nullptr`.

#### **Why It’s Used**
- Iterative search is efficient and avoids the overhead of recursive function calls.

#### **Example**
Search for `3` in the tree:
```
    5
   / \
  3   7
```
- Start at `5`: `3 < 5`, move left to `3`.
- Found `3`, return the node.

---

### **6. Deletion Function**
```cpp
btreeNode* BSTRemove(btreeNode** Parent_Root, int target)
{
    btreeNode *pVRoot = (btreeNode*) malloc (sizeof(btreeNode));
    initbtreeNode(0,*Parent_Root);
    btreeNode* pNode = pVRoot;
    btreeNode* Child_Node = *Parent_Root; 
    btreeNode* dNode;
    ChangeRightTree(*Parent_Root,pVRoot);
    while (Child_Node != nullptr && GetData(Child_Node) != target)
    {
        pNode = Child_Node; 
        if (target < GetData(Child_Node))
            Child_Node = GetLeftTree(Child_Node);
        else
            Child_Node = GetRightTree(Child_Node); 
    }
    if (Child_Node == nullptr)
        return nullptr;
    dNode = Child_Node; 
    if (GetLeftTree(dNode) == nullptr && GetRightTree(dNode) == nullptr)
    {
        if (GetLeftTree(pNode) == dNode)
            RemoveLeftTree(pNode);
        else
            RemoveRightTree(pNode); 
    }
    else if (GetLeftTree(dNode) == nullptr || GetRightTree(dNode) == nullptr)
    {
        btreeNode* dcNode; 
        if (GetLeftTree(dNode) != nullptr)
            dcNode = GetLeftTree(dNode);
        else 
            dcNode = GetRightTree(dNode);
        if (GetLeftTree(pNode) == dNode)
            ChangeLeftTree(dcNode,pNode);
        else
            ChangeRightTree(dcNode,pNode);
    }
    else
    {
        btreeNode* mNode = GetRightTree(dNode); 
        btreeNode* mpNode = dNode; 
        int delData; 
        while (GetLeftTree(mNode) != nullptr) {
            mpNode = mNode;
            mNode = GetLeftTree(mNode); 
        }
        delData = GetData(dNode); 
        SetData(GetData(mNode),dNode);
        if (GetLeftTree(mpNode) == mNode)
            ChangeLeftTree(GetRightTree(mNode),mpNode);
        else
            ChangeRightTree(GetRightTree(mNode),mpNode);
        dNode = mNode; 
        SetData(delData,dNode);
    }
    if (GetRightTree(pVRoot) != *Parent_Root)
        *Parent_Root = GetRightTree(pVRoot);
    free(pVRoot);
    Rebalance(Parent_Root);
    return dNode; 
}
```

#### **What It Does**
- Removes a node with the given `target` value from the BST. Handles three cases:
  1. Node has no children.
  2. Node has one child.
  3. Node has two children.

#### **Breakdown**
1. **Virtual Root**:
   - Create a virtual root (`pVRoot`) to simplify edge cases (e.g., removing the root node).

2. **Search for Target**:
   - Traverse the tree to find the node to delete (`dNode`).

3. **Deletion Cases**:
   - **No Children**: Simply remove the node.
   - **One Child**: Replace the node with its child.
   - **Two Children**: Replace the node with its in-order successor.

4. **Rebalancing**:
   - Call `Rebalance` to maintain the AVL property.

#### **Why It’s Used**
- Deletion in a BST is complex because it must handle multiple cases while maintaining the BST property. The virtual root simplifies edge cases.

#### **Example**
Remove `5` from the tree:
```
    5
   / \
  3   7
```
- Replace `5` with its in-order successor (`7`):
```
    7
   /
  3
```

---

### **Summary**
This code implements a **Binary Search Tree** with **AVL balancing**. It provides functions for initialization, insertion, search, and deletion, ensuring efficient operations and a balanced tree. Each function is carefully designed to handle edge cases and maintain the BST and AVL properties. Let me know if you’d like further clarification or improvements!