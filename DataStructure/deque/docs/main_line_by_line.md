# Step-by-Step Explanation: main.c

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. By the end, you’ll understand every line of code, the logic behind it, and why certain approaches are used.

---

### **1. Header Files**
```c
#include <stdio.h>
#include <stdlib.h>
```
- **What it does**: These lines include standard C libraries.
  - `stdio.h`: Provides functions for input and output, like `printf`.
  - `stdlib.h`: Provides functions for memory management, like `malloc` and `free`.
- **Why it’s used**: These libraries are essential for basic operations like printing to the console and dynamically allocating memory.

---

### **2. Data Structures**
#### **Node Structure**
```c
typedef struct Node
{
    struct Node* prev;
    struct Node* next;
    int data; 
} NODE;
```
- **What it does**: Defines a `NODE` structure for the doubly linked list.
  - `prev`: A pointer to the previous node in the list.
  - `next`: A pointer to the next node in the list.
  - `data`: An integer value stored in the node.
- **Why it’s used**: A doubly linked list allows traversal in both directions (forward and backward), which is essential for a deque.

#### **Deque Structure**
```c
typedef struct Deque
{
    NODE* head;
    NODE* rear;
} DEQUE;
```
- **What it does**: Defines a `DEQUE` structure to represent the deque.
  - `head`: A pointer to the first node in the deque.
  - `rear`: A pointer to the last node in the deque.
- **Why it’s used**: The `head` and `rear` pointers allow efficient access to both ends of the deque, enabling fast insertions and removals.

---

### **3. Helper Functions**
#### **`createnode` Function**
```c
NODE* createnode(int data)
{
    NODE* newNode = (NODE*)malloc(sizeof(NODE));
    newNode->data = data;
    return newNode;
}
```
- **What it does**: Creates a new node with the given `data`.
  - `malloc(sizeof(NODE))`: Allocates memory for a new node.
  - `newNode->data = data`: Assigns the `data` value to the node.
- **Why it’s used**: This function encapsulates the creation of a node, making the code modular and reusable.

#### **`initdeque` Function**
```c
DEQUE* initdeque()
{
    DEQUE* dq = (DEQUE*)malloc(sizeof(DEQUE));
    dq->head = NULL;
    dq->rear = NULL;
    return dq;
}
```
- **What it does**: Initializes an empty deque.
  - `malloc(sizeof(DEQUE))`: Allocates memory for the deque structure.
  - `dq->head = NULL` and `dq->rear = NULL`: Sets both `head` and `rear` to `NULL`, indicating an empty deque.
- **Why it’s used**: This function ensures that the deque is properly initialized before use.

---

### **4. Core Operations**
#### **`push_front` Function**
```c
void push_front(DEQUE * dq, int data)
{
    NODE* newNode = createnode(data);
    newNode->prev = NULL;
    if (dq->head == NULL)
    {
        dq->head = dq->rear = newNode;
        return;
    }
    dq->head->prev = newNode;
    newNode->next = dq->head;
    dq->head = newNode;
}
```
- **What it does**: Inserts a new node at the front of the deque.
  - **Step 1**: Create a new node with the given `data`.
  - **Step 2**: If the deque is empty (`dq->head == NULL`), set both `head` and `rear` to the new node.
  - **Step 3**: If the deque is not empty:
    - Set the `prev` pointer of the current `head` to the new node.
    - Set the `next` pointer of the new node to the current `head`.
    - Update `head` to point to the new node.
- **Why it’s used**: This function allows efficient insertion at the front of the deque.

#### **`push_back` Function**
```c
void push_back(DEQUE * dq, int data)
{
    NODE* newNode = createnode(data);
    newNode->next = NULL;
    if (dq->rear == NULL)
    {
        dq->rear = dq->head = newNode;
        return;
    }
    dq->rear->next = newNode;
    newNode->prev = dq->rear;
    dq->rear = newNode;
}
```
- **What it does**: Inserts a new node at the back of the deque.
  - **Step 1**: Create a new node with the given `data`.
  - **Step 2**: If the deque is empty (`dq->rear == NULL`), set both `head` and `rear` to the new node.
  - **Step 3**: If the deque is not empty:
    - Set the `next` pointer of the current `rear` to the new node.
    - Set the `prev` pointer of the new node to the current `rear`.
    - Update `rear` to point to the new node.
- **Why it’s used**: This function allows efficient insertion at the back of the deque.

#### **`pop_front` Function**
```c
int pop_front(DEQUE* dq)
{
    if (dq->head == NULL)
    {
        return -1;
    }
    NODE* tmp = dq->head;
    int result = tmp->data; 
    dq->head = dq->head->next;
    free(tmp);
    return result; 
}
```
- **What it does**: Removes and returns the node at the front of the deque.
  - **Step 1**: Check if the deque is empty (`dq->head == NULL`). If so, return `-1`.
  - **Step 2**: Store the current `head` node in a temporary variable `tmp`.
  - **Step 3**: Store the `data` of the `head` node in `result`.
  - **Step 4**: Update `head` to point to the next node.
  - **Step 5**: Free the memory of the removed node (`tmp`).
  - **Step 6**: Return the `data` of the removed node.
- **Why it’s used**: This function allows efficient removal from the front of the deque.

#### **`pop_back` Function**
```c
int pop_back(DEQUE* dq)
{
    if (dq->rear == NULL)
    {
        return -1;
    }
    NODE* tmp = dq->rear;
    int result = tmp->data; 
    dq->rear = dq->rear->prev;
    free(tmp);
    return result; 
}
```
- **What it does**: Removes and returns the node at the back of the deque.
  - **Step 1**: Check if the deque is empty (`dq->rear == NULL`). If so, return `-1`.
  - **Step 2**: Store the current `rear` node in a temporary variable `tmp`.
  - **Step 3**: Store the `data` of the `rear` node in `result`.
  - **Step 4**: Update `rear` to point to the previous node.
  - **Step 5**: Free the memory of the removed node (`tmp`).
  - **Step 6**: Return the `data` of the removed node.
- **Why it’s used**: This function allows efficient removal from the back of the deque.

---

### **5. Main Function**
```c
int main()
{
    DEQUE* dq = initdeque();
    for (int i = 0; i < 10; i++)
    {
        push_front(dq, i);
    }
    for (int i = 10; i < 20; i++)
    {
        push_back(dq, i);
    }
    for (int i = 0; i < 20; i++)
    {
        printf("pop front=%d\n", pop_front(dq));
    }
    for (int i = 0; i < 10; i++)
    {
        push_front(dq, i);
    }
    for (int i = 10; i < 20; i++)
    {
        push_back(dq, i);
    }
    for (int i = 0; i < 20; i++)
    {
        printf("pop back= %d\n", pop_back(dq));
    }
    return 0;
}
```
- **What it does**: Demonstrates the use of the deque.
  - **Step 1**: Initialize an empty deque.
  - **Step 2**: Insert numbers `0` to `9` at the front of the deque.
  - **Step 3**: Insert numbers `10` to `19` at the back of the deque.
  - **Step 4**: Remove and print all elements from the front.
  - **Step 5**: Insert numbers `0` to `9` at the front and `10` to `19` at the back again.
  - **Step 6**: Remove and print all elements from the back.
- **Why it’s used**: This function tests the deque implementation and demonstrates its functionality.

---

### **6. Diagrams**
#### **Doubly Linked List Structure**
```
head -> [prev|data|next] <-> [prev|data|next] <-> ... <-> [prev|data|next] <- rear
```
- Each node points to the previous and next nodes, allowing traversal in both directions.

#### **Deque Operations**
- **`push_front`**:
  ```
  Before: head -> [A] <-> [B] <- rear
  After:  head -> [new] <-> [A] <-> [B] <- rear
  ```
- **`push_back`**:
  ```
  Before: head -> [A] <-> [B] <- rear
  After:  head -> [A] <-> [B] <-> [new] <- rear
  ```

---

This concludes the detailed explanation of the code. Let me know if you’d like further clarification or improvements!