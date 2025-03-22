# Step-by-Step Explanation: main.c

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll explain what each part does, why it’s used, and how it fits into the overall program. I’ll also define technical terms and use examples to make everything clear.

---

### **1. Header Files and Includes**
```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
```
- **What it does**: These lines include standard libraries for input/output (`stdio.h`), memory management (`stdlib.h`), and thread management (`pthread.h`).
- **Why it’s used**:
  - `stdio.h` is needed for functions like `printf` to display output.
  - `stdlib.h` is used for dynamic memory allocation (`malloc`, `free`).
  - `pthread.h` provides functions for creating and managing threads, mutexes, and condition variables.

---

### **2. Data Structures**
#### **Node Structure**
```c
typedef struct Node
{
    int data;
    struct Node* next;
    struct Node* prev; 
} node;
```
- **What it does**: Defines a `node` structure for a **doubly linked list**.
  - `data`: Stores the value of the node.
  - `next`: Points to the next node in the list.
  - `prev`: Points to the previous node in the list.
- **Why it’s used**:
  - A doubly linked list allows efficient insertion and removal from both ends of the list (front and back).
  - The `prev` pointer makes it easy to traverse the list backward.

#### **Deque Structure**
```c
typedef struct Deque
{
    int data;
    int sz;
    node* front;
    node* back;
    pthread_mutex_t mutex;
    pthread_cond_t cond; 
} deque;
```
- **What it does**: Defines a `deque` structure to represent the double-ended queue.
  - `data`: Unused in this implementation (likely a leftover or placeholder).
  - `sz`: Tracks the number of elements in the deque.
  - `front`: Points to the first node in the deque.
  - `back`: Points to the last node in the deque.
  - `mutex`: A **mutex** (mutual exclusion lock) to ensure thread safety.
  - `cond`: A **condition variable** to signal threads when data is available.
- **Why it’s used**:
  - The `mutex` ensures that only one thread can modify the deque at a time.
  - The `cond` allows threads to wait for data to be added to the deque.

#### **ThreadParams Structure**
```c
typedef struct {
    deque* param1;
    int param2;
} ThreadParams;
```
- **What it does**: Defines a structure to pass parameters to threads.
  - `param1`: A pointer to the deque.
  - `param2`: An integer value (data to be pushed into the deque).
- **Why it’s used**:
  - Threads in C can only take a single `void*` argument. This structure allows passing multiple parameters to a thread.

---

### **3. Helper Functions**
#### **createThreadParams**
```c
ThreadParams* createThreadParams(deque* d, int data) {
    ThreadParams* params = (ThreadParams*)malloc(sizeof(ThreadParams));
    if (params == NULL) {
        perror("Failed to allocate ThreadParams");
        exit(EXIT_FAILURE);
    }
    params->param1 = d;
    params->param2 = data;
    return params;
}
```
- **What it does**: Allocates memory for a `ThreadParams` struct and initializes it with the given deque and data.
- **Why it’s used**:
  - Ensures that thread parameters are dynamically allocated and properly initialized.
  - Handles memory allocation errors gracefully using `perror` and `exit`.

#### **initdeque**
```c
deque* initdeque() {
    deque* d = (deque*) malloc(sizeof(deque));
    d->front = NULL;
    d->back = NULL;
    d->sz = 0;
    pthread_mutex_init(&d->mutex, NULL);
    pthread_cond_init(&d->cond, NULL);
    return d;
}
```
- **What it does**: Initializes a new deque.
  - Allocates memory for the deque.
  - Sets `front` and `back` to `NULL` (empty deque).
  - Initializes the mutex and condition variable.
- **Why it’s used**:
  - Prepares the deque for use by setting up its initial state and synchronization primitives.

#### **destroydeque**
```c
void destroydeque(deque* d) {
    pthread_cond_destroy(&d->cond);
    pthread_mutex_destroy(&d->mutex);
    free(d);
}
```
- **What it does**: Cleans up the deque by destroying the mutex, condition variable, and freeing its memory.
- **Why it’s used**:
  - Prevents memory leaks and ensures proper cleanup of resources.

---

### **4. Deque Operations**
#### **push_front**
```c
void push_front(void* arg) {
    ThreadParams* params = (ThreadParams*)arg;
    deque* d = params->param1; 
    int data = params->param2;
    pthread_mutex_lock(&d->mutex);
    node* newNode = (node*) malloc(sizeof(node)); 
    newNode->data = data;
    newNode->prev = NULL;
    d->sz++;
    if (d->sz == 1) {
        newNode->next = NULL;
        d->front = d->back = newNode;
    } else {
        d->front->prev = newNode; 
        newNode->next = d->front;
        d->front = newNode;
    }
    pthread_mutex_unlock(&d->mutex);
}
```
- **What it does**: Adds a new node to the front of the deque.
  - Locks the mutex to ensure thread safety.
  - Creates a new node and sets its `data`.
  - Updates the `front` pointer and links the new node to the existing list.
  - Unlocks the mutex when done.
- **Why it’s used**:
  - Allows threads to safely add elements to the front of the deque.

#### **push_back**
```c
void push_back(void* arg) {
    ThreadParams* params = (ThreadParams*)arg; 
    pthread_mutex_lock(&params->param1->mutex);
    deque* d = params->param1; 
    int data = params->param2;
    node* newNode = (node*) malloc(sizeof(node)); 
    newNode->data = data;
    newNode->next = NULL;
    d->sz++;
    if (d->sz == 1) {
        newNode->prev = NULL;
        d->front = d->back = newNode;
    } else {
        d->back->next = newNode; 
        newNode->prev = d->back;
        d->back = newNode;
    }
    pthread_mutex_unlock(&d->mutex);
    pthread_cond_signal(&d->cond);
}
```
- **What it does**: Adds a new node to the back of the deque.
  - Similar to `push_front`, but updates the `back` pointer instead.
  - Signals the condition variable (`cond`) to notify waiting threads.
- **Why it’s used**:
  - Allows threads to safely add elements to the back of the deque and notify other threads.

---

### **5. Main Function**
#### **Thread Creation**
```c
int main() {
    deque* d = initdeque();
    pthread_t threads[NUM_THREADS];
    pthread_t push_threads[NUM_THREADS];
    ThreadParams params[NUM_THREADS];
    for (int i = 0; i < 19; i++) {
        ThreadParams* params = createThreadParams(d, i);
        pthread_create(&push_threads[i], NULL, &push_back, params);
        pthread_cond_wait(&d->cond, &d->mutex);
    }
    pthread_mutex_unlock(&d->mutex);
    for (int i = 0; i < 19; i++) {
        pthread_join(push_threads[i], NULL);
    }
    for (int i = 0; i < 19; i++) {
        pthread_create(&threads[i], NULL, (void*(*)(void*))&pop_back, (void*)d);
    }
    for (int i = 0; i < 19; i++) {
        pthread_join(threads[i], NULL);
    }
    destroydeque(d);
    return 0;
}
```
- **What it does**:
  1. Initializes the deque.
  2. Creates threads to perform `push_back` operations.
  3. Waits for all `push_back` threads to complete.
  4. Creates threads to perform `pop_back` operations.
  5. Waits for all `pop_back` threads to complete.
  6. Destroys the deque.
- **Why it’s used**:
  - Demonstrates how to use the deque in a multi-threaded environment.
  - Ensures proper synchronization and cleanup.

---

### **Text-Based Diagram of Deque**
```
front -> [node1] <-> [node2] <-> [node3] <- back
```
- Each node points to the next and previous nodes.
- `front` points to the first node, and `back` points to the last node.

---

This explanation should make the code accessible to everyone, from beginners to experts! Let me know if you’d like further clarification on any part.