# Step-by-Step Explanation: main.c

Let’s break down the code **step by step** in a way that’s easy to understand, even for someone who is just starting to learn programming. I’ll explain each section in detail, define technical terms, and use examples and diagrams to make everything clear.

---

### **1. Struct Definitions**
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

#### **What It Does**
- These lines define the building blocks of the Binary Search Tree (BST):
  - A `Node` represents a single element in the tree.
  - A `Tree` represents the entire tree, starting from its root.

#### **Explanation**
1. **`Node` Struct:**
   - A `Node` has three parts:
     - `left`: A pointer to the left child node.
     - `right`: A pointer to the right child node.
     - `data`: The integer value stored in the node.
   - Think of a `Node` as a box that holds a number (`data`) and has two arrows (`left` and `right`) pointing to other boxes (child nodes).

2. **`Tree` Struct:**
   - A `Tree` has one part:
     - `root`: A pointer to the first node in the tree (the topmost node).
   - The `root` is like the starting point of the tree. From here, you can follow the `left` and `right` pointers to explore the entire tree.

#### **Why This Structure?**
- The `Node` and `Tree` structs are used to represent the BST in memory. This structure allows us to:
  - Dynamically create and link nodes.
  - Easily traverse the tree using pointers.

---

### **2. Tree Initialization**
```c
tree* inittree()
{
    tree* newTree = (tree*)malloc(sizeof(tree));
    newTree->root = NULL;
    return newTree; 
}
```

#### **What It Does**
- This function creates a new, empty tree.

#### **Explanation**
1. **`malloc(sizeof(tree))`:**
   - `malloc` is a function that allocates memory dynamically (at runtime).
   - `sizeof(tree)` calculates the amount of memory needed to store a `tree` struct.
   - The result is a pointer to the newly allocated memory.

2. **`newTree->root = NULL`:**
   - The `root` of the new tree is set to `NULL`, meaning the tree is empty (it has no nodes yet).

3. **Returning the Tree:**
   - The function returns a pointer to the newly created tree.

#### **Why This Approach?**
- Dynamic memory allocation allows the tree to grow as needed, without requiring a fixed size upfront.
- Setting `root` to `NULL` ensures the tree starts empty.

---

### **3. Inserting into the BST**
```c
void insertBST(tree* tr, int data)
{
    node* newNode = (node*)malloc(sizeof(node));
    newNode->data = data;
    newNode->left = NULL;
    newNode->right = NULL; 

    if (tr->root == NULL)
    {
        tr->root = newNode;
        return;
    }

    node* cNode = tr->root;
    node* pNode;

    while (cNode != NULL)
    {
        pNode = cNode;
        if (cNode->data < data)
        {
            cNode = cNode->right; 
        }
        else
        {
            cNode = cNode->left;
        }
    }

    if (pNode->data < data)
    {
        pNode->right = newNode; 
    }
    else
    {
        pNode->left = newNode;
    }
}
```

#### **What It Does**
- This function inserts a new value into the BST while maintaining the BST property.

#### **Explanation**
1. **Creating a New Node:**
   - A new node is created using `malloc`.
   - Its `data` is set to the value being inserted.
   - Its `left` and `right` pointers are set to `NULL` (it has no children yet).

2. **Handling an Empty Tree:**
   - If the tree is empty (`tr->root == NULL`), the new node becomes the root.

3. **Finding the Correct Position:**
   - Two pointers are used:
     - `cNode` (current node): Starts at the root and moves down the tree.
     - `pNode` (parent node): Keeps track of the parent of `cNode`.
   - The `while` loop traverses the tree:
     - If the new value is greater than `cNode->data`, move to the right child.
     - Otherwise, move to the left child.
   - The loop stops when `cNode` becomes `NULL` (we’ve found the correct position).

4. **Linking the New Node:**
   - After the loop, `pNode` is the parent of the new node.
   - If the new value is greater than `pNode->data`, the new node becomes the right child.
   - Otherwise, it becomes the left child.

#### **Why This Approach?**
- The BST property ensures that the tree remains sorted, making it efficient for searching, insertion, and deletion.
- The use of `cNode` and `pNode` allows us to traverse the tree and find the correct position for the new node.

---

### **4. DFS Traversal (In-Order)**
```c
void printTree(node* NODE)
{
    if (NODE == NULL)
        return;
    printTree(NODE->left);
    printf("%d\n", NODE->data);
    printTree(NODE->right);
}
```

#### **What It Does**
- This function prints the tree in ascending order using an in-order DFS traversal.

#### **Explanation**
1. **Base Case:**
   - If the current node (`NODE`) is `NULL`, the function returns (stops recursing).

2. **Recursive Steps:**
   - `printTree(NODE->left)`: Recursively print the left subtree.
   - `printf("%d\n", NODE->data)`: Print the current node’s value.
   - `printTree(NODE->right)`: Recursively print the right subtree.

#### **Why This Approach?**
- In-order traversal ensures that nodes are printed in ascending order, which is useful for displaying sorted data.

---

### **5. BFS Traversal**
```c
void bfsprint(node *root) {
    if (root == NULL)
        return;
    node* queue[100];  // Queue of node pointers
    int head = 0, tail = 0;
    queue[tail++] = root;
    while (head < tail) {
        node* current = queue[head++];
        printf("%d\n", current->data);
        if (current->left != NULL)
            queue[tail++] = current->left;
        if (current->right != NULL)
            queue[tail++] = current->right;
    }
}
```

#### **What It Does**
- This function prints the tree level by level using BFS traversal.

#### **Explanation**
1. **Queue Setup:**
   - A queue (implemented as an array) is used to store nodes.
   - `head` and `tail` are indices for the queue.

2. **Enqueue the Root:**
   - The root node is added to the queue.

3. **Processing the Queue:**
   - The `while` loop continues until the queue is empty.
   - For each node:
     - Print its value.
     - Add its left and right children to the queue (if they exist).

#### **Why This Approach?**
- BFS traversal is useful for visualizing the tree level by level.

---

### **6. Main Function**
```c
int main()
{
    tree* tr = inittree();
    insertBST(tr, 10);
    insertBST(tr, 12);
    insertBST(tr, 9);
    insertBST(tr, 13);
    bfsprint(tr->root);
}
```

#### **What It Does**
- This function demonstrates the usage of the BST by:
  - Creating a tree.
  - Inserting values.
  - Printing the tree using BFS.

#### **Explanation**
1. **Tree Initialization:**
   - `inittree()` creates an empty tree.

2. **Inserting Values:**
   - `insertBST()` adds values to the tree.

3. **Printing the Tree:**
   - `bfsprint()` prints the tree level by level.

---

### **Summary**
This code is a complete implementation of a Binary Search Tree. It demonstrates:
- How to create and manage a BST.
- How to insert values while maintaining the BST property.
- How to traverse and print the tree using DFS and BFS.

By breaking down each part and explaining the logic, we’ve made the code accessible to everyone, from beginners to experts!