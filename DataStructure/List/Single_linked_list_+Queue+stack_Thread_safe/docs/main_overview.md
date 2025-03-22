# Code Overview: main.c

This C code is a multi-threaded implementation of a **linked list** that can function both as a **stack** and a **queue**. It uses **pthreads** (POSIX threads) to manage concurrent operations on the list, ensuring thread safety through **mutexes** and **condition variables**. Below is a detailed explanation of the purpose, functionality, and structure of the code:

---

### **Purpose of the Code**
The code demonstrates how to implement a **thread-safe linked list** that supports two common data structures:
1. **Stack**: A Last-In-First-Out (LIFO) structure where elements are added and removed from the top.
2. **Queue**: A First-In-First-Out (FIFO) structure where elements are added at the tail and removed from the head.

The program uses **multi-threading** to simulate concurrent operations (push and pop) on the list, ensuring that these operations are synchronized and do not lead to race conditions or data corruption.

---

### **Main Functionality**
1. **Linked List Implementation**:
   - The linked list is implemented using two structures:
     - `Node`: Represents a single element in the list, containing an integer `data` and a pointer to the next node.
     - `List`: Represents the entire list, containing:
       - `sz`: The size of the list.
       - `top`: A pointer to the first node (used for stack operations).
       - `tail`: A pointer to the last node (used for queue operations).
       - `mtx`: A mutex to ensure thread-safe access to the list.
       - `cond`: A condition variable to signal changes in the list (e.g., when a new element is added).

2. **Thread-Safe Operations**:
   - The code uses **mutexes** (`pthread_mutex_t`) to lock critical sections of code (e.g., when modifying the list).
   - **Condition variables** (`pthread_cond_t`) are used to signal threads when the list is updated (e.g., when a new element is pushed).

3. **Multi-Threading**:
   - The program creates two threads:
     - One for testing **queue operations**.
     - One for testing **stack operations**.
   - These threads run concurrently, and the main thread waits for them to complete using `pthread_join`.

4. **Stack and Queue Operations**:
   - **Stack Push (`push_front`)**:
     - Adds a new node to the **front** of the list (LIFO behavior).
     - Updates the `top` pointer and signals other threads using a condition variable.
   - **Queue Push (`push_back`)**:
     - Adds a new node to the **end** of the list (FIFO behavior).
     - Updates the `tail` pointer and signals other threads using a condition variable.

---

### **Algorithms and Data Structures**
1. **Linked List**:
   - A dynamic data structure where each element (node) contains data and a pointer to the next node.
   - The list supports both stack and queue operations by maintaining `top` and `tail` pointers.

2. **Thread Synchronization**:
   - **Mutexes**: Ensure that only one thread can modify the list at a time.
   - **Condition Variables**: Allow threads to wait for specific conditions (e.g., waiting for the list to be non-empty).

3. **Multi-Threading**:
   - The program uses POSIX threads (`pthread_t`) to create and manage threads.
   - Threads are synchronized using `pthread_mutex_lock`, `pthread_mutex_unlock`, and `pthread_cond_wait`.

---

### **Overall Structure**
1. **Global Variables**:
   - `NUM_THREADS`: Defines the number of threads (not fully utilized in the provided code).
   - `majormtx` and `majorcond`: Global mutex and condition variable for synchronizing the main thread with other threads.

2. **List Initialization**:
   - The `initlist` function initializes a new list, setting its size to 0 and initializing its mutex and condition variable.

3. **Thread Parameters**:
   - The `ThreadParams` structure is used to pass parameters (list and data) to thread functions.

4. **Thread Functions**:
   - `push_front`: Implements stack push.
   - `push_back`: Implements queue push.
   - `queueThreadtest` and `stackThreadtest` (not fully implemented in the provided code): These would contain the logic for testing queue and stack operations.

5. **Main Function**:
   - Creates two threads (`queuethread` and `stackthread`).
   - Uses a global mutex and condition variable to synchronize the creation of these threads.
   - Waits for the threads to complete using `pthread_join`.

---

### **Problem Being Solved**
The code addresses the challenge of **managing shared data structures in a multi-threaded environment**. Specifically, it ensures that:
- Multiple threads can safely modify a linked list without causing race conditions.
- Threads can wait for specific conditions (e.g., waiting for the list to be updated) without busy-waiting.

---

### **How the Parts Work Together**
1. The `main` function initializes the program and creates threads.
2. Each thread performs operations (push) on the shared list.
3. Mutexes and condition variables ensure that:
   - Only one thread modifies the list at a time.
   - Threads are notified when the list is updated.
4. The main thread waits for all worker threads to complete before exiting.

---

### **Key Takeaways**
- The code demonstrates **thread-safe programming** using mutexes and condition variables.
- It shows how a single data structure (linked list) can support multiple abstract data types (stack and queue).
- The use of multi-threading allows for concurrent operations, which is essential in real-world applications like web servers or databases.

This code is a great example of how to handle shared resources in a multi-threaded environment while maintaining data integrity and synchronization. However, it is incomplete (e.g., `queueThreadtest` and `stackThreadtest` are not fully implemented), and there are areas for improvement, which we can discuss in the next questions.