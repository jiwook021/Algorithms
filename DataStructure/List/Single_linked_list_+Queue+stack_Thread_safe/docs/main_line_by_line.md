# Step-by-Step Explanation: main.c

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple language, define technical terms, and provide examples to make everything clear. We’ll also explore the **why** behind the code’s design choices.

---

### **1. Header Files and Global Variables**
```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
```
- **What it does**: These are **header files** that provide necessary functions and definitions.
  - `stdio.h`: For input/output functions like `printf`.
  - `stdlib.h`: For memory allocation functions like `malloc`.
  - `pthread.h`: For multi-threading functions like `pthread_create`.
  - `unistd.h`: For system calls like `sleep`.

```c
const int NUM_THREADS = 10;
static pthread_mutex_t majormtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t majorcond = PTHREAD_COND_INITIALIZER;
```
- **What it does**:
  - `NUM_THREADS`: A constant defining the number of threads (not fully used in this code).
  - `majormtx`: A **mutex** (short for "mutual exclusion") used to synchronize access to shared resources.
  - `majorcond`: A **condition variable** used to signal threads when a condition is met (e.g., when the list is updated).

- **Why it’s used**:
  - Mutexes prevent multiple threads from accessing shared resources simultaneously, which could lead to **race conditions** (unpredictable behavior).
  - Condition variables allow threads to wait for specific events (e.g., waiting for the list to be non-empty).

---

### **2. Node and List Structures**
```c
typedef struct Node {
    int data;
    struct Node* next; 
} node; 
```
- **What it does**:
  - Defines a `Node` structure, which represents a single element in the linked list.
  - Each node contains:
    - `data`: The value stored in the node (an integer).
    - `next`: A pointer to the next node in the list.

- **Why it’s used**:
  - A linked list is a dynamic data structure where each element (node) points to the next one. This allows for efficient insertion and deletion.

```c
typedef struct List {
    int sz;    
    node* top;
    node* tail;
    pthread_mutex_t mtx;
    pthread_cond_t cond;
} list;
```
- **What it does**:
  - Defines a `List` structure, which represents the entire linked list.
  - It contains:
    - `sz`: The size of the list (number of nodes).
    - `top`: A pointer to the first node (used for stack operations).
    - `tail`: A pointer to the last node (used for queue operations).
    - `mtx`: A mutex to synchronize access to the list.
    - `cond`: A condition variable to signal changes in the list.

- **Why it’s used**:
  - The `List` structure encapsulates all the information needed to manage the linked list, including synchronization mechanisms for multi-threading.

---

### **3. List Initialization**
```c
list* initlist() {
    list* l = (list*)malloc(sizeof(list));
    l->sz = 0; 
    l->top = NULL;
    pthread_mutex_init(&l->mtx, NULL);
    pthread_cond_init(&l->cond, NULL);
    return l;
}
```
- **What it does**:
  - Allocates memory for a new list using `malloc`.
  - Initializes the list’s size (`sz`) to 0 and its `top` pointer to `NULL` (empty list).
  - Initializes the mutex and condition variable.

- **Why it’s used**:
  - This function ensures that the list is properly set up before any operations are performed on it.

---

### **4. Thread Parameters Structure**
```c
typedef struct {
    list* param1;
    int param2;
} ThreadParams;
```
- **What it does**:
  - Defines a structure to pass parameters to thread functions.
  - `param1`: A pointer to the list.
  - `param2`: An integer value (e.g., data to be added to the list).

- **Why it’s used**:
  - Thread functions in C can only take one argument (`void*`), so this structure allows passing multiple parameters.

---

### **5. Stack Push Operation**
```c
void push_front(void* arg) {
    ThreadParams* params = (ThreadParams*)arg;
    list* l = params->param1;
    int data = params->param2;
    node* newNode = (node*)malloc(sizeof(node));
    newNode->data = data; 
    l->sz++;
    if(l->top == NULL) {
        l->top = l->tail = newNode;
        newNode->next = NULL;
        pthread_mutex_lock(&l->mtx);
        pthread_cond_signal(&l->cond);
        pthread_mutex_unlock(&l->mtx);
        return;
    }
    newNode->next = l->top;
    l->top = newNode;
    pthread_mutex_lock(&l->mtx);
    pthread_cond_signal(&l->cond);
    pthread_mutex_unlock(&l->mtx);  
    return;
}
```
- **What it does**:
  - Adds a new node to the **front** of the list (stack behavior).
  - If the list is empty, the new node becomes both the `top` and `tail`.
  - Otherwise, the new node is inserted at the front, and the `top` pointer is updated.

- **Why it’s used**:
  - This implements the **LIFO** (Last-In-First-Out) behavior of a stack.

- **Control Flow**:
  1. Extract parameters from `arg`.
  2. Allocate memory for a new node.
  3. Update the list size.
  4. If the list is empty, set the new node as both `top` and `tail`.
  5. Otherwise, insert the new node at the front.
  6. Signal other threads using the condition variable.

---

### **6. Main Function**
```c
int main() {
    pthread_t queuethread, stackthread;
    pthread_create(&queuethread, NULL, &queueThreadtest, NULL);    
    pthread_mutex_lock(&majormtx);
    pthread_cond_wait(&majorcond, &majormtx);
    pthread_mutex_unlock(&majormtx);
    pthread_create(&stackthread, NULL, &stackThreadtest, NULL);
    if(pthread_join(queuethread, NULL) != 0) {
        perror("pthread_join");  
    }
    if(pthread_join(stackthread, NULL) != 0) {
        perror("pthread_join");  
    }
    return 0;
}
```
- **What it does**:
  - Creates two threads: one for queue operations and one for stack operations.
  - Uses a global mutex and condition variable to synchronize thread creation.
  - Waits for both threads to complete using `pthread_join`.

- **Why it’s used**:
  - Demonstrates how to create and manage threads in a multi-threaded program.

---

### **Text-Based Diagram of Linked List**
```
Empty List:
top -> NULL
tail -> NULL

After push_front(10):
top -> [10|next] -> NULL
tail -> [10|next] -> NULL

After push_front(20):
top -> [20|next] -> [10|next] -> NULL
tail -> [10|next] -> NULL
```

This breakdown should make the code accessible to everyone, from beginners to experts! Let me know if you’d like to dive deeper into any specific part.