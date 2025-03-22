# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple language, examples, and diagrams to make everything clear.

---

### **1. Includes and Dependencies**
```cpp
#include <iostream>
#include <thread>
#include <semaphore>
#include <chrono>
#include <vector>
#include <mutex>
```
- **What it does**: These lines include necessary libraries for the program.
  - `<iostream>`: For input/output (e.g., printing to the console).
  - `<thread>`: For creating and managing threads.
  - `<semaphore>`: For synchronization between threads.
  - `<chrono>`: For time-related operations (e.g., delays).
  - `<vector>`: For dynamic arrays (used in the circular queue).
  - `<mutex>`: For mutual exclusion (though not used in this code).

- **Why it’s used**: These libraries provide the tools needed for threading, synchronization, and data storage.

---

### **2. CircularQueue Class**
```cpp
class CircularQueue {
public:
    explicit CircularQueue(size_t size)
        : m_size(size), m_head(0), m_tail(0), m_count(0), m_data(size) {}
```
- **What it does**: Defines a circular queue, a fixed-size buffer that wraps around when full.
  - `m_size`: Maximum size of the queue.
  - `m_head`: Index of the front of the queue (where items are removed).
  - `m_tail`: Index of the back of the queue (where items are added).
  - `m_count`: Number of items currently in the queue.
  - `m_data`: A vector (dynamic array) to store the items.

- **Why it’s used**: A circular queue efficiently manages a fixed-size buffer by reusing space after items are removed.

---

#### **Enqueue Method**
```cpp
bool enqueue(int item) {
    if (m_count == m_size) {
        return false; // Queue full
    }
    m_data[m_tail] = item;
    m_tail = (m_tail + 1) % m_size;
    ++m_count;
    return true;
}
```
- **What it does**: Adds an item to the queue.
  - Checks if the queue is full (`m_count == m_size`).
  - If full, returns `false` (item cannot be added).
  - If not full:
    - Stores the item at the `m_tail` index.
    - Moves `m_tail` forward, wrapping around using modulo (`%`).
    - Increments `m_count`.

- **Example**:
  - If `m_size = 5`, `m_tail = 4`, and `m_count = 4`:
    - `m_tail = (4 + 1) % 5 = 0` (wraps around to the start).

- **Why it’s used**: Ensures items are added efficiently without reallocating memory.

---

#### **Dequeue Method**
```cpp
int dequeue(int& item) {
    if (m_count == 0) {
        return -1; // Queue empty
    }
    item = m_data[m_head];
    m_head = (m_head + 1) % m_size;
    --m_count;
    return item;
}
```
- **What it does**: Removes an item from the queue.
  - Checks if the queue is empty (`m_count == 0`).
  - If empty, returns `-1` (no item to remove).
  - If not empty:
    - Retrieves the item at the `m_head` index.
    - Moves `m_head` forward, wrapping around using modulo (`%`).
    - Decrements `m_count`.

- **Example**:
  - If `m_size = 5`, `m_head = 4`, and `m_count = 5`:
    - `m_head = (4 + 1) % 5 = 0` (wraps around to the start).

- **Why it’s used**: Ensures items are removed efficiently without reallocating memory.

---

### **3. SafeQueue Class**
```cpp
class SafeQueue {
public:
    explicit SafeQueue(size_t maxSize)
        : queue_(maxSize), m_availableItems(0), m_availableSlots(maxSize) {}
```
- **What it does**: Wraps the circular queue with thread-safe synchronization.
  - `queue_`: The underlying circular queue.
  - `m_availableItems`: A semaphore tracking available items for consumers.
  - `m_availableSlots`: A semaphore tracking available slots for producers.

- **Why it’s used**: Ensures thread-safe access to the queue using semaphores.

---

#### **Produce Method**
```cpp
void produce(int item) {
    m_availableSlots.acquire(); // wait if no slot available
    {
        queue_.enqueue(item);
        std::cout << "Produced: " << item << "\n";
    }
    m_availableItems.release(); // signal item available
}
```
- **What it does**: Adds an item to the queue in a thread-safe way.
  - Waits for an available slot (`m_availableSlots.acquire()`).
  - Adds the item to the queue (`queue_.enqueue(item)`).
  - Signals that a new item is available (`m_availableItems.release()`).

- **Why it’s used**: Ensures producers wait when the queue is full and signal when they add an item.

---

#### **Consume Method**
```cpp
void consume() {
    m_availableItems.acquire(); // wait if no item available
    int item;
    std::cout << "Consumed: " << queue_.dequeue(item) << "\n";
    m_availableSlots.release(); // signal slot available
}
```
- **What it does**: Removes an item from the queue in a thread-safe way.
  - Waits for an available item (`m_availableItems.acquire()`).
  - Removes the item from the queue (`queue_.dequeue(item)`).
  - Signals that a new slot is available (`m_availableSlots.release()`).

- **Why it’s used**: Ensures consumers wait when the queue is empty and signal when they remove an item.

---

### **4. Producer and Consumer Functions**
```cpp
void producer(SafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.produce(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
```
- **What it does**: Simulates a producer generating items.
  - Loops `numItems` times, producing items (`queue.produce(i)`).
  - Adds a delay (`sleep_for`) to simulate real-world processing.

- **Why it’s used**: Demonstrates how producers add items to the queue.

---

```cpp
void consumer(SafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.consume();
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
}
```
- **What it does**: Simulates a consumer processing items.
  - Loops `numItems` times, consuming items (`queue.consume()`).
  - Adds a delay (`sleep_for`) to simulate real-world processing.

- **Why it’s used**: Demonstrates how consumers remove items from the queue.

---

### **5. Main Function**
```cpp
int main() {
    constexpr int queueSize = 5;
    constexpr int numItems = 20;

    SafeQueue queue(queueSize);
    std::thread producerThread(producer, std::ref(queue), numItems);
    std::thread consumerThread(consumer, std::ref(queue), numItems);

    producerThread.join();
    consumerThread.join();

    return 0;
}
```
- **What it does**: Initializes the program.
  - Creates a `SafeQueue` with a fixed size (`queueSize`).
  - Starts producer and consumer threads.
  - Waits for threads to finish (`join`).

- **Why it’s used**: Coordinates the entire program and ensures threads complete before exiting.

---

### **Diagram: Circular Queue**
```
Initial State:
Index: 0 1 2 3 4
Value: - - - - -
Head: 0, Tail: 0, Count: 0

After Enqueue(10):
Index: 0 1 2 3 4
Value: 10 - - - -
Head: 0, Tail: 1, Count: 1

After Enqueue(20):
Index: 0 1 2 3 4
Value: 10 20 - - -
Head: 0, Tail: 2, Count: 2

After Dequeue():
Index: 0 1 2 3 4
Value: - 20 - - -
Head: 1, Tail: 2, Count: 1
```

---

### **Summary**
This code demonstrates:
- A **circular queue** for efficient buffer management.
- **Semaphores** for thread-safe synchronization.
- **Producer-consumer threads** for concurrent processing.

Each part is carefully designed to ensure thread safety, avoid race conditions, and efficiently manage shared resources.