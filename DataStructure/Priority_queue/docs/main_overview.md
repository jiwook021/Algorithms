# Code Overview: main.c

This C code implements a **Priority Queue** using a **Max-Heap** data structure. Let's break down the purpose, functionality, and structure of the code in detail.

---

### **Purpose of the Code**
The purpose of this code is to implement a **Priority Queue**, which is a data structure that allows efficient insertion and removal of elements based on their priority. In this implementation, the priority queue is designed such that the element with the **highest value** has the highest priority. This is achieved using a **Max-Heap**, a binary tree where the value of each node is greater than or equal to the values of its children.

The code provides the following operations:
1. **Insertion (`push`)**: Adds a new element to the priority queue while maintaining the heap property.
2. **Removal (`pop`)**: Removes and returns the element with the highest priority (the maximum value in the heap).
3. **Peek (`top`)**: Returns the element with the highest priority without removing it.
4. **Check if empty (`isEmpty`)**: Determines whether the priority queue is empty.

The code also includes utility functions to maintain the heap structure, such as `siftUp` and `siftDown`, which ensure that the heap property is preserved after insertions and deletions.

---

### **Main Functionality**
The code solves the problem of efficiently managing a collection of elements where the highest-priority element (the maximum value) needs to be accessed or removed quickly. This is a common requirement in algorithms like Dijkstra's shortest path, scheduling tasks, or any scenario where elements need to be processed in order of priority.

The **Max-Heap** is implemented using an array, where:
- The root of the heap is at index `0`.
- For any node at index `i`:
  - Its left child is at index `2*i + 1`.
  - Its right child is at index `2*i + 2`.
  - Its parent is at index `(i - 1) / 2`.

The heap property ensures that the value of each node is greater than or equal to the values of its children.

---

### **Algorithms Used**
1. **Heap Operations**:
   - **`siftUp`**: Ensures that a newly inserted element moves up the heap to its correct position to maintain the heap property.
   - **`siftDown`**: Ensures that the root element (after removal) moves down the heap to its correct position to maintain the heap property.

2. **Priority Queue Operations**:
   - **`push`**: Inserts a new element into the heap and uses `siftUp` to maintain the heap property.
   - **`pop`**: Removes the root element (the maximum value), replaces it with the last element in the heap, and uses `siftDown` to maintain the heap property.
   - **`top`**: Returns the root element without modifying the heap.
   - **`isEmpty`**: Checks if the heap is empty.

---

### **Overall Structure**
The code is organized into the following components:
1. **Data Structure**:
   - A `PriorityQueue` struct is defined, which contains:
     - An array `data` to store the elements of the heap.
     - An integer `size` to track the number of elements in the heap.

2. **Utility Functions**:
   - `parent`, `left`, and `right`: Helper functions to calculate the indices of a node's parent and children in the array-based heap.

3. **Heap Maintenance Functions**:
   - `siftUp`: Moves an element up the heap to restore the heap property after insertion.
   - `siftDown`: Moves an element down the heap to restore the heap property after removal.

4. **Priority Queue Operations**:
   - `initializePriorityQueue`: Initializes the priority queue with a size of `0`.
   - `push`: Inserts a new element into the priority queue.
   - `pop`: Removes and returns the highest-priority element.
   - `top`: Returns the highest-priority element without removing it.
   - `isEmpty`: Checks if the priority queue is empty.

5. **Main Function**:
   - Demonstrates the usage of the priority queue by inserting elements (`3`, `5`, `1`, `4`) and then removing and printing them in descending order (`5`, `4`, `3`, `1`).

---

### **How the Parts Work Together**
1. The `PriorityQueue` struct stores the heap data and its size.
2. The utility functions (`parent`, `left`, `right`) help navigate the heap structure.
3. The `siftUp` and `siftDown` functions maintain the heap property after insertions and deletions.
4. The `push` function inserts a new element and uses `siftUp` to ensure the heap property is preserved.
5. The `pop` function removes the root element, replaces it with the last element, and uses `siftDown` to restore the heap property.
6. The `top` function provides access to the highest-priority element without modifying the heap.
7. The `isEmpty` function checks if the heap is empty.
8. The `main` function demonstrates the usage of the priority queue by inserting elements and printing them in descending order.

---

### **Problem Being Solved**
The code solves the problem of efficiently managing a dynamic set of elements where the highest-priority element needs to be accessed or removed quickly. This is achieved using a Max-Heap, which ensures that both insertion and removal operations have a time complexity of **O(log n)**, where `n` is the number of elements in the heap.

---

### **Approach Taken**
The approach taken is to use an array-based Max-Heap to implement the priority queue. This approach is efficient because:
- The array provides a compact representation of the heap.
- The `siftUp` and `siftDown` operations ensure that the heap property is maintained with logarithmic time complexity.
- The priority queue operations (`push`, `pop`, `top`, `isEmpty`) are simple and intuitive to use.

---

### **Summary**
This code implements a priority queue using a Max-Heap, providing efficient insertion, removal, and access to the highest-priority element. The heap is maintained using `siftUp` and `siftDown` operations, and the priority queue is demonstrated in the `main` function by inserting and removing elements in descending order. This implementation is both efficient and easy to understand, making it a great example of how to use heaps to solve priority-based problems.