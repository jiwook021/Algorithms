# Code Overview: main.cpp

This C++ code implements a **thread-safe producer-consumer pattern** using a **circular queue** and **semaphores**. Let’s break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The code solves the classic **producer-consumer problem**, which is a common synchronization challenge in concurrent programming. The problem involves two types of threads:
1. **Producer threads**: Generate data and add it to a shared buffer (queue).
2. **Consumer threads**: Remove and process data from the shared buffer.

The goal is to ensure that:
- Producers don’t add data to a full buffer.
- Consumers don’t try to remove data from an empty buffer.
- Access to the shared buffer is thread-safe (no race conditions).

The code achieves this using a **circular queue** for efficient buffer management and **semaphores** for synchronization.

---

### **Main Functionality**
1. **Circular Queue**:
   - A fixed-size buffer that wraps around when it reaches the end.
   - Efficiently manages space by reusing slots after items are consumed.
   - Provides `enqueue` (add) and `dequeue` (remove) operations.

2. **Thread-Safe Queue (SafeQueue)**:
   - Wraps the circular queue with synchronization mechanisms.
   - Uses **semaphores** to track available slots and items.
   - Ensures producers and consumers wait when the queue is full or empty.

3. **Producer and Consumer Threads**:
   - Producer threads generate items and add them to the queue.
   - Consumer threads remove items from the queue and process them.
   - Both threads run concurrently, with delays to simulate real-world processing times.

4. **Synchronization**:
   - Semaphores (`m_availableItems` and `m_availableSlots`) ensure proper coordination between producers and consumers.
   - A mutex (`m_mutex`) is declared but not used in this implementation (it could be added for additional thread safety if needed).

---

### **Algorithms and Data Structures**
1. **Circular Queue**:
   - Uses a `std::vector<int>` to store data.
   - Tracks the head (front of the queue), tail (back of the queue), and count (number of items).
   - Implements modular arithmetic (`%`) to wrap around the buffer.

2. **Semaphores**:
   - `m_availableItems`: Tracks the number of items available for consumption.
   - `m_availableSlots`: Tracks the number of empty slots available for production.
   - Semaphores ensure that producers wait when the queue is full and consumers wait when the queue is empty.

3. **Threading**:
   - Uses `std::thread` to create producer and consumer threads.
   - Threads run concurrently, with delays to simulate real-world processing.

---

### **Overall Structure**
The code is organized into several components:
1. **CircularQueue Class**:
   - Manages the underlying buffer.
   - Provides `enqueue` and `dequeue` methods.

2. **SafeQueue Class**:
   - Wraps the circular queue with synchronization.
   - Provides `produce` and `consume` methods for thread-safe operations.

3. **Producer and Consumer Functions**:
   - Simulate data production and consumption.
   - Use `SafeQueue` to interact with the shared buffer.

4. **Main Function**:
   - Initializes the queue and starts producer and consumer threads.
   - Waits for threads to complete using `join`.

---

### **How the Parts Work Together**
1. **Initialization**:
   - The `main` function creates a `SafeQueue` with a fixed size.
   - Producer and consumer threads are started with references to the queue.

2. **Producer Thread**:
   - Calls `produce` to add items to the queue.
   - Waits if the queue is full (using `m_availableSlots`).
   - Signals when an item is added (using `m_availableItems`).

3. **Consumer Thread**:
   - Calls `consume` to remove items from the queue.
   - Waits if the queue is empty (using `m_availableItems`).
   - Signals when a slot is freed (using `m_availableSlots`).

4. **Synchronization**:
   - Semaphores ensure that producers and consumers don’t access the queue simultaneously in an unsafe way.
   - The circular queue efficiently manages the buffer space.

---

### **Problem Being Solved**
The code solves the **producer-consumer problem**, which arises when multiple threads share a common resource (the queue). Without proper synchronization:
- Producers might overwrite data in a full queue.
- Consumers might read invalid data from an empty queue.
- Race conditions could corrupt the queue’s state.

The solution ensures:
- Thread-safe access to the queue.
- Proper coordination between producers and consumers.
- Efficient use of buffer space.

---

### **Approach Taken**
1. **Circular Queue**:
   - Provides efficient buffer management with O(1) enqueue and dequeue operations.
   - Avoids unnecessary memory reallocation.

2. **Semaphores**:
   - Ensure producers and consumers wait when necessary.
   - Provide a lightweight synchronization mechanism compared to mutexes.

3. **Threading**:
   - Simulates real-world scenarios where producers and consumers operate at different speeds.

---

### **Summary**
This code demonstrates a robust solution to the producer-consumer problem using:
- A **circular queue** for efficient buffer management.
- **Semaphores** for synchronization.
- **Threads** for concurrent execution.

It ensures thread safety, avoids race conditions, and efficiently manages shared resources. This pattern is widely used in real-world applications like message queues, task schedulers, and streaming systems.