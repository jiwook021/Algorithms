# Step-by-Step Explanation: main.c

Let’s dive into the code step by step, breaking it down into manageable pieces and explaining everything in detail. I’ll start from the top and work our way down, ensuring that every line is clear and understandable.

---

### **1. Header Files**
```c
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
```
- **What it does**: These lines include standard C libraries that provide essential functions.
  - `stdio.h`: Provides input/output functions like `printf` (used to print text to the console).
  - `stdlib.h`: Provides memory allocation functions like `malloc` (used to allocate memory dynamically) and `free` (used to deallocate memory).
  - `memory.h`: Provides memory manipulation functions (though not used in this code).
- **Why it’s used**: These libraries are necessary for the program to perform basic operations like printing to the console and managing memory.

---

### **2. Data Structures**
#### **NODE Structure**
```c
typedef struct node
{
    int data; 
    struct node* next;
    struct node* previous;
} NODE;
```
- **What it does**: Defines a structure called `NODE` that represents a single element in the queue.
  - `data`: Stores the value of the node (e.g., an integer like 10 or 20).
  - `next`: A pointer to the next node in the queue.
  - `previous`: A pointer to the previous node in the queue.
- **Why it’s used**: This structure is the building block of the doubly linked list. Each node holds a value and links to its neighbors, allowing the queue to grow dynamically.

#### **Queue Structure**
```c
typedef struct queue 
{
    int size;
    NODE* front;
    NODE* back;
} Queue;
```
- **What it does**: Defines a structure called `Queue` that represents the queue itself.
  - `size`: Tracks the number of elements in the queue.
  - `front`: A pointer to the first node in the queue.
  - `back`: A pointer to the last node in the queue.
- **Why it’s used**: This structure keeps track of the queue’s state, including its size and the locations of its front and back elements.

---

### **3. Queue Initialization**
```c
Queue* initqueue()
{
    Queue* q = (Queue*) malloc(sizeof(Queue));
    q->size = 0;
    q->front = NULL;
    q->back = NULL;
    return q;
}
```
- **What it does**: Initializes an empty queue.
  - `malloc(sizeof(Queue))`: Allocates memory for a new `Queue` structure.
  - `q->size = 0`: Sets the size of the queue to 0 (empty).
  - `q->front = NULL` and `q->back = NULL`: Sets the front and back pointers to `NULL` (no nodes yet).
- **Why it’s used**: This function prepares the queue for use by allocating memory and setting initial values.

---

### **4. Queue Operations**
#### **size()**
```c
int size(struct queue* q)
{
    return q->size; 
}
```
- **What it does**: Returns the number of elements in the queue.
- **Why it’s used**: Provides a way to check how many elements are currently in the queue.

#### **empty()**
```c
int empty(struct queue* q)
{
    return q->size ? 0 : 1;   
}
```
- **What it does**: Checks if the queue is empty.
  - `q->size ? 0 : 1`: If `q->size` is not 0, returns 0 (false); otherwise, returns 1 (true).
- **Why it’s used**: Provides a way to check if the queue has no elements.

#### **front()**
```c
int front(struct queue* q)
{
    return q->size ? q->front->data : 0; 
}
```
- **What it does**: Returns the value of the front element in the queue.
  - If the queue is not empty (`q->size` is not 0), returns `q->front->data` (the value of the front node).
  - If the queue is empty, returns 0.
- **Why it’s used**: Provides a way to access the front element without removing it.

#### **back()**
```c
int back(struct queue* q)
{
    return q->size ? q->back->data : 0; 
}
```
- **What it does**: Returns the value of the back element in the queue.
  - If the queue is not empty (`q->size` is not 0), returns `q->back->data` (the value of the back node).
  - If the queue is empty, returns 0.
- **Why it’s used**: Provides a way to access the back element without removing it.

---

### **5. Node Initialization**
```c
NODE* initnode(int data)
{
    NODE* newNode = (NODE*)malloc(sizeof(NODE));
    newNode->next = NULL;
    newNode->previous = NULL;
    newNode->data = data;
    return newNode;
}
```
- **What it does**: Creates and initializes a new node with the given data.
  - `malloc(sizeof(NODE))`: Allocates memory for a new `NODE`.
  - `newNode->next = NULL` and `newNode->previous = NULL`: Sets the `next` and `previous` pointers to `NULL` (no neighbors yet).
  - `newNode->data = data`: Sets the node’s value to the provided `data`.
- **Why it’s used**: This function prepares a new node for insertion into the queue.

---

### **6. Push Operation**
```c
void push(Queue* q, int data)
{   
    NODE* node = initnode(data);
    q->size++;
    if(q->front == NULL || q->back == NULL)
    {
        q->back = node;
        q->front = node; 
        printf("empty in push function q->back = %d\n", q->back->data);
        return;
    }
    node->previous = q->back; 
    q->back->next = node; 
    q->back = node; 
    printf("in push function q->back = %d\n", q->back->data);
}
```
- **What it does**: Adds a new node to the back of the queue.
  - `initnode(data)`: Creates a new node with the given data.
  - `q->size++`: Increments the queue size.
  - If the queue is empty (`q->front` or `q->back` is `NULL`), the new node becomes both the front and back.
  - Otherwise, the new node is linked to the current back node, and the back pointer is updated.
- **Why it’s used**: This function adds elements to the queue, maintaining the FIFO order.

---

### **7. Pop Operation**
```c
int pop(Queue* q)
{
    if(empty(q))
    {
        printf("Queue is empty!");
    }
    NODE* temp = q->front;
    int item = temp->data;
    q->front = q->front->next;
    
    if (q->front == NULL) {
        q->back = NULL;
    } else {
        q->front->previous = NULL;
    }  
    free(temp);
    q->size--;
    return item;
}
```
- **What it does**: Removes and returns the front element of the queue.
  - If the queue is empty, prints a message.
  - Otherwise, removes the front node, updates the front pointer, and frees the memory.
- **Why it’s used**: This function removes elements from the queue, maintaining the FIFO order.

---

### **8. Freeing the Queue**
```c
void freeQueue(Queue* queue) {
    while (!empty(queue)) {
        printf("Freed:%d!\n", pop(queue));
    }
    free(queue);
}
```
- **What it does**: Frees all memory allocated for the queue.
  - Repeatedly calls `pop()` to remove and free nodes until the queue is empty.
  - Frees the `Queue` structure itself.
- **Why it’s used**: Ensures no memory leaks occur when the queue is no longer needed.

---

### **9. Main Function**
```c
int main()
{
    Queue* queue = initqueue();
    push(queue, 10);
    push(queue, 20);
    push(queue, 30);
    push(queue, 40);
    printf("Dequeued item = %d\n", pop(queue));
    printf("Front item is %d\n", front(queue));
    printf("Rear item is %d\n", back(queue));
    freeQueue(queue);
    return 0; 
}
```
- **What it does**: Demonstrates the usage of the queue.
  - Initializes a queue.
  - Adds elements (10, 20, 30, 40).
  - Removes and prints the front element.
  - Prints the front and back elements.
  - Frees the queue.
- **Why it’s used**: Shows how the queue functions work in practice.

---

### **Text-Based Diagram of the Queue**
```
Initial State:
Queue: [ ] (empty)

After push(10):
Queue: [10]

After push(20):
Queue: [10] <-> [20]

After push(30):
Queue: [10] <-> [20] <-> [30]

After push(40):
Queue: [10] <-> [20] <-> [30] <-> [40]

After pop():
Queue: [20] <-> [30] <-> [40]
```

---

This concludes the step-by-step explanation. Let me know if you’d like further clarification on any part!