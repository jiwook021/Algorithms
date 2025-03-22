# Step-by-Step Explanation: SparseMatrix.c

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple language, examples, and diagrams to make everything clear, even for someone who is just starting to learn programming.

---

### **1. Header Files and Includes**
```c
#include "SparseMatrix.h"
#include "LinkedList.h"
#include <stdio.h>
#include <stdlib.h>
```

#### What it does:
- These lines include necessary header files for the program to work.
  - `SparseMatrix.h` and `LinkedList.h` are custom header files that define the `SparseMatrix` and `LinkedList` structures and their associated functions.
  - `<stdio.h>` is the standard input/output library, used for printing to the console.
  - `<stdlib.h>` is the standard library, used for memory allocation (e.g., `malloc`).

#### Why it’s used:
- Header files allow us to organize code into reusable modules. For example:
  - `SparseMatrix.h` defines what a sparse matrix is and what operations it supports.
  - `LinkedList.h` defines how linked lists work, which are used to store the non-empty elements of the matrix.

---

### **2. Initializing the Sparse Matrix (`InitMatrix`)**
```c
void InitMatrix(SparseMatrix* mat, int height, int width)
{
    mat->wList = (LinkedList*)malloc(sizeof(LinkedList) * width);

    for (int i = 0; i < width; i++)
    {
        InitList((mat->wList) + i);
    }
    mat->hList = (LinkedList*)malloc(sizeof(LinkedList) * height);
    for (int i = 0; i < height; i++)
    {
        InitList((mat->hList) + i);
    }
}
```

#### What it does:
- This function initializes a sparse matrix by setting up its row and column linked lists.

#### Step-by-step breakdown:
1. **Allocate memory for column linked lists (`wList`)**:
   - `mat->wList = (LinkedList*)malloc(sizeof(LinkedList) * width);`
     - `malloc` allocates memory for an array of `LinkedList` structures. The size of the array is `width`, which is the number of columns in the matrix.
     - Each element in this array will store a linked list representing a column of the matrix.

2. **Initialize each column linked list**:
   - `for (int i = 0; i < width; i++) { InitList((mat->wList) + i); }`
     - This loop goes through each column and initializes its linked list using the `InitList` function (likely defined in `LinkedList.h`).
     - `(mat->wList) + i` is a pointer to the `i`-th column’s linked list.

3. **Allocate memory for row linked lists (`hList`)**:
   - `mat->hList = (LinkedList*)malloc(sizeof(LinkedList) * height);`
     - Similar to the columns, this allocates memory for an array of `LinkedList` structures. The size of the array is `height`, which is the number of rows in the matrix.

4. **Initialize each row linked list**:
   - `for (int i = 0; i < height; i++) { InitList((mat->hList) + i); }`
     - This loop initializes each row’s linked list.

#### Why it’s used:
- A sparse matrix is represented using two arrays of linked lists:
  - `wList`: One linked list for each column.
  - `hList`: One linked list for each row.
- This dual representation allows efficient access to both rows and columns, which is crucial for operations like insertion and deletion.

#### Example:
If we have a 3x3 matrix:
```
Row 0: [A, 0, 0]
Row 1: [0, B, 0]
Row 2: [0, 0, C]
```
- `hList` will store:
  - Row 0: A
  - Row 1: B
  - Row 2: C
- `wList` will store:
  - Column 0: A
  - Column 1: B
  - Column 2: C

---

### **3. Inserting an Element into the Matrix (`MatInsert`)**
```c
void MatInsert(SparseMatrix* mat, char data, int row, int col)
{
    Node* newNode = (Node*)malloc(sizeof(Node));
    LNInsert(&(mat->hList[row]), newNode, data);
    LBInsert(&(mat->wList[col]), newNode, data);
}
```

#### What it does:
- This function inserts a new element (`data`) into the matrix at a specific row and column.

#### Step-by-step breakdown:
1. **Create a new node**:
   - `Node* newNode = (Node*)malloc(sizeof(Node));`
     - Allocates memory for a new node to store the element.

2. **Insert into the row linked list**:
   - `LNInsert(&(mat->hList[row]), newNode, data);`
     - Inserts the new node into the linked list for the specified row (`hList[row]`).

3. **Insert into the column linked list**:
   - `LBInsert(&(mat->wList[col]), newNode, data);`
     - Inserts the same node into the linked list for the specified column (`wList[col]`).

#### Why it’s used:
- The same node is inserted into both the row and column linked lists. This ensures that the matrix remains consistent and that the element can be accessed efficiently from both its row and column.

#### Example:
If we insert `'X'` at row 1, column 2:
- `hList[1]` (Row 1) will now include `'X'`.
- `wList[2]` (Column 2) will now include `'X'`.

---

### **4. Deleting an Element from the Matrix (`MatDelete`)**
```c
void MatDelete(SparseMatrix* mat, char data, int row)
{
    for (int i = 0; i <= row; i++)
    {
        NodeDelete(&(mat->hList[i]), data);
    }
}
```

#### What it does:
- This function deletes all occurrences of a specific element (`data`) from the matrix up to a specified row.

#### Step-by-step breakdown:
1. **Loop through rows**:
   - `for (int i = 0; i <= row; i++)`
     - This loop goes through each row from 0 to the specified row.

2. **Delete the element from each row**:
   - `NodeDelete(&(mat->hList[i]), data);`
     - Calls the `NodeDelete` function to remove all nodes containing `data` from the `i`-th row’s linked list.

#### Why it’s used:
- This function ensures that the matrix is cleaned up by removing unwanted elements. It only deletes up to the specified row to limit the scope of the operation.

---

### **5. Printing the Matrix (`printMatrix`, `printRow`, `printCol`)**
```c
void printMatrix(SparseMatrix* mat, int row, int col)
{
    mat->hList[row].current = mat->hList[row].head;

    printf("Matrix\n\n");

    for (int j = 1; j <= row; j++)
    {
        printf("\nrow %d: ", j);

        for (int i = 0; i < mat->hList[j].size; i++)
        {
            printf("%c ", mat->hList[j].current->data);
            mat->hList[j].current = mat->hList[j].current->next;
        }
    }
    printf("\n");
}
```

#### What it does:
- This function prints the contents of the matrix by traversing the row linked lists.

#### Step-by-step breakdown:
1. **Set the current pointer**:
   - `mat->hList[row].current = mat->hList[row].head;`
     - Sets the `current` pointer to the head of the linked list for the specified row.

2. **Print the matrix header**:
   - `printf("Matrix\n\n");`

3. **Loop through rows**:
   - `for (int j = 1; j <= row; j++)`
     - This loop goes through each row from 1 to the specified row.

4. **Print each row**:
   - `for (int i = 0; i < mat->hList[j].size; i++)`
     - This loop traverses the linked list for the current row and prints each element.

#### Why it’s used:
- This function provides a way to visualize the contents of the matrix, which is useful for debugging and understanding the data structure.

---

### **6. Printing Rows and Columns (`printRow`, `printCol`)**
```c
void printRow(SparseMatrix* mat, int row)
{
    mat->hList[row].current = mat->hList[row].head;

    printf("Matrix row %d: ", row);

    for (int i = 0; i < mat->hList[row].size; i++)
    {
        printf("%c ", mat->hList[row].current->data);
        mat->hList[row].current = mat->hList[row].current->next;
    }
    printf("\n");
}

void printCol(SparseMatrix* mat, int col)
{
    mat->wList[col].current = mat->wList[col].head;

    printf("Matrix column %d: ", col);

    for (int i = 0; i < mat->wList[col].size; i++)
    {
        printf("%c ", mat->wList[col].current->data);
        mat->wList[col].current = mat->wList[col].current->below;
    }

    printf("\n");
}
```

#### What it does:
- These functions print the contents of a specific row or column.

#### Step-by-step breakdown:
1. **Set the current pointer**:
   - For rows: `mat->hList[row].current = mat->hList[row].head;`
   - For columns: `mat->wList[col].current = mat->wList[col].head;`

2. **Print the row/column header**:
   - For rows: `printf("Matrix row %d: ", row);`
   - For columns: `printf("Matrix column %d: ", col);`

3. **Traverse and print elements**:
   - For rows: Traverse the row linked list using `next`.
   - For columns: Traverse the column linked list using `below`.

#### Why it’s used:
- These functions allow you to inspect specific rows or columns, which is useful for debugging or analyzing the matrix.

---

### **Summary of Key Concepts**
1. **Sparse Matrix**:
   - A matrix where most elements are empty (or zero). Instead of storing all elements, we only store the non-empty ones.

2. **Linked Lists**:
   - A data structure where each element (node) points to the next element. Here, each row and column is represented as a linked list.

3. **Dual Representation**:
   - The matrix is stored using two arrays of linked lists: one for rows and one for columns. This allows efficient access to both rows and columns.

4. **Dynamic Memory Allocation**:
   - Memory is allocated at runtime using `malloc`, allowing the matrix to grow and shrink as needed.

---

### **Diagram of Sparse Matrix Representation**
```
Row 0: A -> 0 -> 0
Row 1: 0 -> B -> 0
Row 2: 0 -> 0 -> C

Column 0: A -> 0 -> 0
Column 1: 0 -> B -> 0
Column 2: 0 -> 0 -> C
```

Each row and column is a linked list, and the same node is shared between the row and column lists. This ensures consistency and efficient access.

---

This concludes the detailed explanation of the code. Let me know if you’d like further clarification on any part!