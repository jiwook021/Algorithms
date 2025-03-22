# Code Overview: LB_queue.cpp

This C++ code implements a **Queue data structure** using a **linked list** approach. Let's break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The purpose of this code is to implement a **Queue**, which is a fundamental data structure in computer science. A queue follows the **First-In-First-Out (FIFO)** principle, meaning the first element added to the queue is the first one to be removed. This is analogous to a real-world queue, such as a line of people waiting for a service.

The code provides the following core functionalities:
1. **Enqueue**: Add an element to the end of the queue.
2. **Dequeue**: Remove and return the element at the front of the queue.
3. **Peek**: View the element at the front of the queue without removing it.
4. **Check if the queue is empty**: Determine whether the queue contains any elements.

The queue is implemented using a **singly linked list**, where each element (node) in the queue contains:
- A `data` field to store the value.
- A `next` pointer to link to the next node in the queue.

---

### **Algorithms and Data Structures Used**
1. **Linked List**:
   - The queue is built using a linked list, which is a dynamic data structure where each node points to the next node.
   - This allows the queue to grow and shrink dynamically as elements are added or removed.

2. **FIFO Principle**:
   - The queue maintains two pointers:
     - `front`: Points to the first node in the queue (the next element to be removed).
     - `rear`: Points to the last node in the queue (the most recently added element).
   - When an element is enqueued, it is added to the `rear`.
   - When an element is dequeued, it is removed from the `front`.

3. **Smart Pointers**:
   - The code uses `std::shared_ptr` to manage memory automatically. This ensures that memory is deallocated when no longer needed, preventing memory leaks.

---

### **Overall Structure**
The code is organized into a class called `Queue`, which encapsulates the queue's functionality. Here's how the different parts of the code work together:

1. **Constructor (`Queue::Queue`)**:
   - Initializes the `front` and `rear` pointers to `nullptr`, indicating an empty queue.

2. **`QisEmpty` Method**:
   - Checks if both `front` and `rear` are `nullptr`. If true, the queue is empty.

3. **`enqueue` Method**:
   - Adds a new element to the end of the queue.
   - Creates a new node using `std::shared_ptr`.
   - Updates the `rear` pointer to point to the new node.
   - If the queue is empty, both `front` and `rear` are set to the new node.

4. **`dequeue` Method**:
   - Removes and returns the element at the front of the queue.
   - Updates the `front` pointer to point to the next node.
   - Returns the data from the removed node.

5. **`peek` Method**:
   - Returns the data of the node at the front of the queue without removing it.

---

### **How the Code Works Together**
1. **Initialization**:
   - When a `Queue` object is created, the constructor initializes `front` and `rear` to `nullptr`, indicating an empty queue.

2. **Enqueue Operation**:
   - When `enqueue` is called, a new node is created and added to the `rear` of the queue.
   - If the queue is empty, both `front` and `rear` point to the new node.
   - Otherwise, the new node is linked to the current `rear`, and `rear` is updated to point to the new node.

3. **Dequeue Operation**:
   - When `dequeue` is called, the node at the `front` is removed, and its data is returned.
   - The `front` pointer is updated to point to the next node.
   - If the queue becomes empty after dequeue, `rear` is also set to `nullptr`.

4. **Peek Operation**:
   - The `peek` method simply returns the data of the node at the `front` without modifying the queue.

5. **Empty Check**:
   - The `QisEmpty` method checks if both `front` and `rear` are `nullptr`, indicating an empty queue.

---

### **Problem Being Solved**
The code solves the problem of managing a collection of elements in a FIFO order. Queues are commonly used in scenarios such as:
- Task scheduling (e.g., printing tasks in a printer queue).
- Breadth-First Search (BFS) in graph algorithms.
- Buffering data in networking or streaming applications.

---

### **Approach Taken**
The code uses a **linked list** to implement the queue because:
- It allows dynamic resizing (unlike arrays, which have a fixed size).
- It efficiently handles enqueue and dequeue operations in **O(1) time complexity**.

The use of `std::shared_ptr` ensures that memory is managed automatically, reducing the risk of memory leaks or dangling pointers.

---

### **Summary**
This code provides a clean and efficient implementation of a queue using a linked list. It demonstrates key concepts such as dynamic memory management, pointers, and the FIFO principle. The structure is modular, with each method performing a specific task, making the code easy to understand and maintain.

Let me know if you'd like to proceed with the next question!