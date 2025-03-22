# Step-by-Step Explanation: main.c

Absolutely! Let’s dive into a **step-by-step, beginner-friendly explanation** of the code. I’ll break it down into sections, explain each part in detail, and use examples and diagrams to make everything crystal clear.

---

### **1. Data Structures: `node` and `tree`**
#### Code:
```c
typedef struct Node 
{
    struct Node* left;
    struct Node* right;
    int data;
} node;

typedef struct Tree
{
    node* root;
} tree;
```

#### Explanation:
1. **What is a `node`?**
   - A `node` is a building block of a **Binary Search Tree (BST)**. Think of it as a container that holds:
     - `data`: The value stored in the node (e.g., an integer like `5`).
     - `left`: A pointer to the **left child node** (smaller values go here).
     - `right`: A pointer to the **right child node** (larger values go here).
   - Example: If a node has `data = 5`, its `left` child might point to a node with `data = 3`, and its `right` child might point to a node with `data = 7`.

2. **What is a `tree`?**
   - A `tree` represents the entire BST. It has:
     - `root`: A pointer to the **topmost node** in the tree. All other nodes are accessed through the `root`.

3. **Why use these structures?**
   - The `node` structure allows us to create a tree where each node can have up to two children, following the BST property.
   - The `tree` structure keeps track of the starting point (`root`) of the tree, making it easy to access and manipulate the entire tree.

---

### **2. Initialization Functions**
#### Code:
```c
tree* initTree()
{
    tree* tr = malloc(sizeof(tree));
    tr->root = NULL;
    return tr;
}

node* initNode(int data)
{
    node* newNode = malloc(sizeof(node));
    newNode->data = data;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}
```

#### Explanation:
1. **`initTree` Function**:
   - **What it does**: Creates an empty tree.
   - **How it works**:
     - `malloc(sizeof(tree))`: Allocates memory for a new `tree` structure.
     - `tr->root = NULL`: Sets the `root` of the tree to `NULL` (no nodes yet).
     - Returns the newly created tree.
   - **Why use `malloc`?**
     - `malloc` dynamically allocates memory on the heap, allowing the tree to exist even after the function ends.

2. **`initNode` Function**:
   - **What it does**: Creates a new node with the given `data`.
   - **How it works**:
     - `malloc(sizeof(node))`: Allocates memory for a new `node`.
     - `newNode->data = data`: Sets the node’s `data` to the provided value.
     - `newNode->left = NULL` and `newNode->right = NULL`: Initializes the `left` and `right` pointers to `NULL` (no children yet).
     - Returns the newly created node.
   - **Why initialize `left` and `right` to `NULL`?**
     - This ensures the node starts as a leaf node (no children), which is the default state for new nodes.

---

### **3. Insertion: `pushBST` Function**
#### Code:
```c
void pushBST(tree* tr, int data)
{
    node* newNode = initNode(data);
    if (tr->root == NULL)
    {
        tr->root = newNode;
        return;
    }
    node* pNode; 
    node* cNode = tr->root; 
    while (cNode != NULL)
    {    
        pNode = cNode;
        if (data > cNode->data)    
        {
            cNode = cNode->right; 
        }
        else
        {
            cNode = cNode->left;
        }
    }
    if (data > pNode->data)    
    {
        pNode->right = newNode; 
    }
    else
    {
        pNode->left = newNode;
    }
}
```

#### Explanation:
1. **What it does**: Inserts a new node with the given `data` into the BST while maintaining the BST property.

2. **Step-by-Step Logic**:
   - **Step 1**: Create a new node using `initNode(data)`.
   - **Step 2**: Check if the tree is empty (`tr->root == NULL`). If it is, set the new node as the `root`.
   - **Step 3**: If the tree is not empty, traverse the tree to find the correct position for the new node:
     - Start at the `root` (`cNode = tr->root`).
     - Use a `while` loop to move down the tree:
       - If `data > cNode->data`, move to the right child (`cNode = cNode->right`).
       - Otherwise, move to the left child (`cNode = cNode->left`).
     - Keep track of the parent node (`pNode`) so you can link the new node to it later.
   - **Step 4**: Once the correct position is found, link the new node to its parent:
     - If `data > pNode->data`, set `pNode->right = newNode`.
     - Otherwise, set `pNode->left = newNode`.

3. **Example**:
   - Suppose the tree has nodes `5`, `3`, and `7`. If you insert `4`:
     - Start at `5`. Since `4 < 5`, move to the left child (`3`).
     - At `3`, since `4 > 3`, move to the right child (which is `NULL`).
     - Link `4` as the right child of `3`.

4. **Why this approach?**
   - The BST property ensures that the tree remains sorted, making searches, insertions, and deletions efficient.

---

### **4. Deletion: `deleteNode` Function**
#### Code:
```c
node* deleteNode(node* current, int data)
{
    if (current == NULL)
        return current;
    if (data > current->data)
        current->right = deleteNode(current->right, data);
    else if (data < current->data)
        current->left = deleteNode(current->left, data);
    else
    {
        if (current->left == NULL)
        {   
            node* temp = current->right;
            free(current);
            return temp;
        }
        else if (current->right == NULL)
        {
            node* temp = current->left;
            free(current);
            return temp;
        }
        node* temp = findMin(current->right);
        current->data = temp->data;
        current->right = deleteNode(current->right, temp->data);
    }
    return current;
}
```

#### Explanation:
1. **What it does**: Deletes a node with the given `data` from the BST while maintaining the BST property.

2. **Step-by-Step Logic**:
   - **Base Case**: If the current node is `NULL`, return `NULL` (nothing to delete).
   - **Recursive Search**:
     - If `data > current->data`, recursively delete from the right subtree.
     - If `data < current->data`, recursively delete from the left subtree.
   - **Node Found**:
     - **Case 1**: The node has no left child. Replace it with its right child.
     - **Case 2**: The node has no right child. Replace it with its left child.
     - **Case 3**: The node has two children:
       - Find the **minimum value in the right subtree** (the smallest value larger than the current node).
       - Replace the current node’s `data` with this minimum value.
       - Recursively delete the duplicate node in the right subtree.

3. **Example**:
   - Suppose the tree has nodes `5`, `3`, and `7`. If you delete `5`:
     - Find the minimum in the right subtree (`7`).
     - Replace `5` with `7` and delete the duplicate `7`.

4. **Why this approach?**
   - Deleting a node with two children requires finding a replacement that maintains the BST property. The minimum value in the right subtree is the smallest value larger than the current node, making it the ideal replacement.

---

### **5. In-Order Traversal: `inorderPrint` Function**
#### Code:
```c
void inorderPrint(node* node)
{
    if (node == NULL)
        return;
    inorderPrint(node->left);
    printf("%d\n", node->data);
    inorderPrint(node->right);
}
```

#### Explanation:
1. **What it does**: Prints the BST in **ascending order** by visiting nodes in the order: left subtree → root → right subtree.

2. **Step-by-Step Logic**:
   - **Base Case**: If the current node is `NULL`, return (nothing to print).
   - **Recursive Calls**:
     - First, recursively print the left subtree.
     - Then, print the current node’s `data`.
     - Finally, recursively print the right subtree.

3. **Example**:
   - For a tree with nodes `5`, `3`, and `7`, the output would be:
     ```
     3
     5
     7
     ```

4. **Why use in-order traversal?**
   - In-order traversal ensures that the nodes are printed in sorted order, which is a key feature of BSTs.

---

### **6. Main Function**
#### Code:
```c
int main()
{
    tree* tr = initTree();
    pushBST(tr, 5);
    pushBST(tr, 4);
    pushBST(tr, 3);
    pushBST(tr, 6);
    pushBST(tr, 8);
    node* rootNode = tr->root;
    inorderPrint(rootNode);
    printf("\n");
    deleteNode(rootNode, 3);
    inorderPrint(rootNode);
    printf("\n");
    deleteNode(rootNode, 5);
    inorderPrint(rootNode);
    printf("\n");   
}
```

#### Explanation:
1. **What it does**: Demonstrates the functionality of the BST by:
   - Creating a tree.
   - Inserting nodes.
   - Printing the tree.
   - Deleting nodes.
   - Printing the tree after each deletion.

2. **Step-by-Step Logic**:
   - **Step 1**: Create an empty tree using `initTree`.
   - **Step 2**: Insert nodes `5`, `4`, `3`, `6`, and `8` using `pushBST`.
   - **Step 3**: Print the tree using `inorderPrint`.
   - **Step 4**: Delete node `3` and print the tree again.
   - **Step 5**: Delete node `5` and print the tree again.

3. **Example Output**:
   ```
   3
   4
   5
   6
   8

   4
   5
   6
   8

   4
   6
   8
   ```

4. **Why this structure?**
   - The `main` function serves as a test harness to demonstrate how the BST works and verify that the insertion, deletion, and traversal functions behave as expected.

---

### **Summary**
This code is a complete implementation of a Binary Search Tree in C. It demonstrates how to:
- Create and initialize a BST.
- Insert nodes while maintaining the BST property.
- Delete nodes while preserving the BST structure.
- Print the tree in sorted order using in-order traversal.

Each part of the code works together to solve the problem of efficiently storing and managing a collection of integers in a sorted manner. If you have any questions or need further clarification, feel free to ask!