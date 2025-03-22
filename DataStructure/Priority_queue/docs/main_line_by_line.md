# Step-by-Step Explanation: main.c

Let’s dive into the code step by step, breaking it down into manageable pieces and explaining everything in detail. I’ll start from the top and work our way down, ensuring that every line is clear and understandable.

---

### **1. Header Files and Constants**
```c
#include <stdio.h>
#include <stdlib.h>
#define MAX_SIZE 100  // define maximum size for the array
```

#### **What it does:**
- `#include <stdio.h>`: This includes the **Standard Input/Output** library, which provides functions like `printf` and `fprintf` for printing to the console or files.
- `#include <stdlib.h>`: This includes the **Standard Library**, which provides functions like `exit` for terminating the program.
- `#define MAX_SIZE 100`: This defines a constant named `MAX_SIZE` with a value of `100`. It represents the maximum number of elements the priority queue can hold.

#### **Why it’s used:**
- The `#include` directives allow us to use pre-built functions for input/output and program control.
- `MAX_SIZE` is used to limit the size of the priority queue, preventing it from growing indefinitely and causing memory issues.

---

### **2. PriorityQueue Struct**
```c
typedef struct {
    int data[MAX_SIZE];
    int size;
} PriorityQueue;
```

#### **What it does:**
- This defines a **struct** (a custom data type) named `PriorityQueue`. It contains:
  - `data[MAX_SIZE]`: An array to store the elements of the priority queue.
  - `size`: An integer to keep track of how many elements are currently in the priority queue.

#### **Why it’s used:**
- The `PriorityQueue` struct bundles the data and its size into a single unit, making it easier to manage and pass around in functions.

#### **Example:**
If we create a `PriorityQueue` object:
```c
PriorityQueue pq;
```
- `pq.data` will hold the elements of the heap.
- `pq.size` will start at `0` and increase as elements are added.

---

### **3. Utility Functions**
```c
int parent(int i) { return (i - 1) / 2; }
int left(int i) { return 2 * i + 1; }
int right(int i) { return 2 * i + 2; }
```

#### **What they do:**
- These functions calculate the indices of a node’s parent and children in the heap:
  - `parent(i)`: Returns the index of the parent of the node at index `i`.
  - `left(i)`: Returns the index of the left child of the node at index `i`.
  - `right(i)`: Returns the index of the right child of the node at index `i`.

#### **Why they’re used:**
- In a heap, the relationships between nodes are defined by their indices in the array. These functions make it easy to navigate the heap structure.

#### **Example:**
For a node at index `3`:
- `parent(3)` returns `1` (since `(3 - 1) / 2 = 1`).
- `left(3)` returns `7` (since `2 * 3 + 1 = 7`).
- `right(3)` returns `8` (since `2 * 3 + 2 = 8`).

---

### **4. siftUp Function**
```c
void siftUp(PriorityQueue* pq, int i) {
    while (i > 0 && pq->data[parent(i)] < pq->data[i]) {
        int temp = pq->data[i];
        pq->data[i] = pq->data[parent(i)];
        pq->data[parent(i)] = temp;
        i = parent(i);
    }
}
```

#### **What it does:**
- This function moves the element at index `i` up the heap until it is in the correct position to maintain the **Max-Heap property** (where the parent is always greater than or equal to its children).

#### **How it works:**
1. It checks if the current node (`i`) is not the root (`i > 0`) and if its parent is smaller than it.
2. If so, it swaps the current node with its parent.
3. It repeats this process until the node is in the correct position.

#### **Why it’s used:**
- After inserting a new element at the end of the heap, `siftUp` ensures the heap property is restored.

#### **Example:**
Suppose we insert `10` into the heap:
- Initially, `10` is at the end of the heap.
- `siftUp` compares `10` with its parent and swaps them if necessary, moving `10` up until it’s in the correct position.

---

### **5. siftDown Function**
```c
void siftDown(PriorityQueue* pq, int i) {
    int maxIndex = i;
    int l = left(i);
    if (l < pq->size && pq->data[l] > pq->data[maxIndex]) {
        maxIndex = l;
    }
    int r = right(i);
    if (r < pq->size && pq->data[r] > pq->data[maxIndex]) {
        maxIndex = r;
    }
    if (i != maxIndex) {
        int temp = pq->data[i];
        pq->data[i] = pq->data[maxIndex];
        pq->data[maxIndex] = temp;
        siftDown(pq, maxIndex);
    }
}
```

#### **What it does:**
- This function moves the element at index `i` down the heap until it is in the correct position to maintain the **Max-Heap property**.

#### **How it works:**
1. It compares the current node with its left and right children.
2. If either child is larger, it swaps the current node with the larger child.
3. It repeats this process recursively until the node is in the correct position.

#### **Why it’s used:**
- After removing the root element, `siftDown` ensures the heap property is restored by moving the new root down to its correct position.

#### **Example:**
Suppose we remove the root (`10`) and replace it with the last element (`3`):
- `siftDown` compares `3` with its children and swaps it with the larger child (`5`), then repeats the process until `3` is in the correct position.

---

### **6. Priority Queue Operations**
#### **initializePriorityQueue**
```c
void initializePriorityQueue(PriorityQueue* pq) {
    pq->size = 0;
}
```
- Initializes the priority queue by setting its size to `0`.

#### **push**
```c
void push(PriorityQueue* pq, int value) {
    if (pq->size == MAX_SIZE) {
        fprintf(stderr, "Priority queue is full\n");
        exit(EXIT_FAILURE);
    }
    pq->data[pq->size] = value;
    siftUp(pq, pq->size);
    pq->size++;
}
```
- Adds a new element to the priority queue and uses `siftUp` to maintain the heap property.

#### **pop**
```c
int pop(PriorityQueue* pq) {
    if (pq->size == 0) {
        fprintf(stderr, "Priority queue is empty\n");
        exit(EXIT_FAILURE);
    }
    int result = pq->data[0];
    pq->size--;
    pq->data[0] = pq->data[pq->size];
    siftDown(pq, 0);
    return result;
}
```
- Removes and returns the highest-priority element (the root) and uses `siftDown` to maintain the heap property.

#### **top**
```c
int top(PriorityQueue* pq) {
    if (pq->size == 0) {
        fprintf(stderr, "Priority queue is empty\n");
        exit(EXIT_FAILURE);
    }
    return pq->data[0];
}
```
- Returns the highest-priority element without removing it.

#### **isEmpty**
```c
int isEmpty(PriorityQueue* pq) {
    return pq->size == 0;
}
```
- Checks if the priority queue is empty.

---

### **7. Main Function**
```c
int main() {
    PriorityQueue pq;
    initializePriorityQueue(&pq);
    push(&pq, 3);
    push(&pq, 5);
    push(&pq, 1);
    push(&pq, 4);
    while (!isEmpty(&pq)) {
        printf("%d ", pop(&pq));
    }
    return 0;
}
```

#### **What it does:**
- Demonstrates the usage of the priority queue by inserting elements (`3`, `5`, `1`, `4`) and then removing and printing them in descending order (`5`, `4`, `3`, `1`).

#### **How it works:**
1. Initializes the priority queue.
2. Inserts elements using `push`.
3. Removes and prints elements using `pop` until the queue is empty.

---

### **Summary**
This code implements a priority queue using a Max-Heap, ensuring efficient insertion and removal of elements. Each function is designed to maintain the heap property, and the `main` function demonstrates how to use the priority queue. By breaking down the code step by step, we can understand how each part contributes to the overall functionality.