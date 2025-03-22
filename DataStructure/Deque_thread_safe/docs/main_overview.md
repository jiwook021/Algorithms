# Code Overview: main.c

This C code implements a **thread-safe double-ended queue (deque)** using a doubly linked list. The deque supports concurrent operations (push and pop from both ends) by multiple threads, ensuring thread safety through the use of **mutexes** and **condition variables**. Let's break down the purpose, functionality, and structure of the code in detail.

---

### **Purpose of the Code**
The code solves the problem of **managing a shared data structure (a deque) in a multi-threaded environment**. In multi-threaded programs, multiple threads may try to access and modify shared data simultaneously, which can lead to **race conditions** (unpredictable behavior due to concurrent access). This code ensures that:
1. The deque is **thread-safe**: Only one thread can modify the deque at a time.
2. Threads can **wait for data** to be available in the deque (using condition variables).
3. The deque supports **efficient insertion and removal** from both ends (front and back).

The code is structured to demonstrate how to:
- Implement a **doubly linked list** for the deque.
- Use **mutexes** to protect shared data.
- Use **condition variables** to synchronize threads.
- Handle **dynamic memory allocation** and **cleanup** properly.

---

### **Main Functionality**
The code provides the following key functionalities:
1. **Deque Operations**:
   - `push_front`: Adds an element to the front of the deque.
   - `push_back`: Adds an element to the back of the deque.
   - `pop_front`: Removes and returns an element from the front of the deque.
   - `pop_back`: Removes and returns an element from the back of the deque.
   - `print_deque`: Prints the contents of the deque.

2. **Thread Safety**:
   - Each operation locks a **mutex** before accessing or modifying the deque.
   - The `push_back` operation signals a **condition variable** (`cond`) to notify waiting threads that new data is available.

3. **Thread Management**:
   - The `main` function creates multiple threads to perform `push_back` and `pop_back` operations concurrently.
   - Threads are synchronized using `pthread_join` to ensure all operations complete before the program exits.

---

### **Algorithms and Data Structures**
1. **Doubly Linked List**:
   - The deque is implemented as a doubly linked list, where each node (`node` struct) contains:
     - `data`: The value stored in the node.
     - `next`: A pointer to the next node in the list.
     - `prev`: A pointer to the previous node in the list.
   - This allows efficient insertion and removal from both ends of the deque.

2. **Mutexes**:
   - A **mutex** (`pthread_mutex_t`) is used to ensure that only one thread can modify the deque at a time.
   - Each operation locks the mutex before accessing the deque and unlocks it afterward.

3. **Condition Variables**:
   - A **condition variable** (`pthread_cond_t`) is used to signal threads when new data is added to the deque.
   - This is particularly useful in scenarios where threads need to wait for data to become available (e.g., in a producer-consumer pattern).

4. **Thread Parameters**:
   - The `ThreadParams` struct is used to pass parameters (the deque and data) to threads.
   - This allows multiple threads to operate on the same deque with different data.

---

### **Overall Structure**
The code is organized into the following components:
1. **Data Structures**:
   - `node`: Represents a single element in the doubly linked list.
   - `deque`: Represents the deque itself, with pointers to the front and back nodes, a size counter, a mutex, and a condition variable.
   - `ThreadParams`: Used to pass parameters to threads.

2. **Initialization and Cleanup**:
   - `initdeque`: Initializes a new deque, including the mutex and condition variable.
   - `destroydeque`: Cleans up the deque, including destroying the mutex and condition variable.

3. **Deque Operations**:
   - `push_front`, `push_back`, `pop_front`, `pop_back`: Implement the core deque operations.
   - `print_deque`: Prints the contents of the deque.

4. **Thread Management**:
   - `createThreadParams`: Allocates memory for thread parameters.
   - `main`: Creates and manages threads to perform deque operations.

---

### **How the Code Works Together**
1. **Initialization**:
   - The `main` function initializes the deque using `initdeque`.

2. **Thread Creation**:
   - The `main` function creates multiple threads to perform `push_back` operations.
   - Each thread is passed a `ThreadParams` struct containing the deque and the data to push.

3. **Thread Synchronization**:
   - After creating the threads, the `main` function waits for them to complete using `pthread_join`.
   - The `push_back` operation signals the condition variable (`cond`) to notify waiting threads.

4. **Deque Operations**:
   - Threads perform `push_back` and `pop_back` operations concurrently, with mutexes ensuring thread safety.
   - The `print_deque` function is used to inspect the contents of the deque.

5. **Cleanup**:
   - After all operations are complete, the `main` function destroys the deque using `destroydeque`.

---

### **Key Takeaways**
- The code demonstrates how to implement a **thread-safe data structure** in C.
- It uses **mutexes** and **condition variables** to handle concurrent access and synchronization.
- The **doubly linked list** provides efficient insertion and removal from both ends of the deque.
- The `main` function orchestrates the creation and synchronization of threads to perform concurrent operations on the deque.

This code is a great example of how to handle shared data in a multi-threaded environment while ensuring correctness and efficiency.