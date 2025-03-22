# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step** in a way that’s easy to understand, even for someone who is just starting to learn programming. I’ll explain each part of the code, define technical terms, and provide examples to make everything clear.

---

### 1. **Header Files and Includes**
```cpp
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <semaphore>
#include <chrono>
#include <optional>
```

#### What It Does:
These lines include libraries that provide functionality for:
- **Input/Output (`<iostream>`)**: Used for printing messages to the console.
- **Queue (`<queue>`)**: Provides a data structure to store items in a First-In-First-Out (FIFO) order.
- **Threading (`<thread>`)**: Allows the program to create and manage multiple threads (parallel tasks).
- **Mutex (`<mutex>`)**: Provides tools for synchronization (not used in this code, but often needed for thread safety).
- **Semaphore (`<semaphore>`)**: A synchronization tool used to control access to shared resources.
- **Chrono (`<chrono>`)**: Used for time-related operations, like adding delays.
- **Optional (`<optional>`)**: Provides a way to represent values that might or might not exist (not used in this code).

#### Why It’s Used:
These libraries are essential for building a multi-threaded program that simulates the Producer-Consumer problem. They provide the tools needed to manage threads, synchronize access to shared resources, and simulate work with delays.

---

### 2. **UnsafeQueue Class**
```cpp
class UnsafeQueue {
public:
    explicit UnsafeQueue()
    : m_availableItems(0) {}
```

#### What It Does:
This defines a class called `UnsafeQueue`. A **class** is a blueprint for creating objects that encapsulate data and behavior. This class represents a shared queue that producers and consumers will use.

- **`explicit UnsafeQueue()`**: This is the constructor for the class. It initializes the `m_availableItems` semaphore to 0, meaning there are no items in the queue initially.
- **`m_availableItems(0)`**: This is a **semaphore** that tracks the number of items in the queue. A semaphore is a synchronization tool that allows threads to wait until a resource is available.

#### Why It’s Used:
The `UnsafeQueue` class is designed to simulate a shared resource (the queue) that multiple threads will access. The semaphore ensures that consumers wait when the queue is empty.

---

### 3. **Produce Function**
```cpp
void produce(int item) {
    queue_.push(item);  // No synchronization, unsafe access
    std::cout << "Produced: " << item << "\n";
    m_availableItems.release();
}
```

#### What It Does:
This function adds an item to the queue.

1. **`queue_.push(item)`**: Adds the `item` to the queue. The queue is a FIFO data structure, so the item is added to the back.
2. **`std::cout << "Produced: " << item << "\n"`**: Prints a message to the console indicating that an item has been produced.
3. **`m_availableItems.release()`**: Increments the semaphore to indicate that a new item is available in the queue.

#### Why It’s Used:
The `produce` function simulates a producer adding items to the queue. The semaphore is incremented to signal consumers that they can now consume an item.

---

### 4. **Consume Function**
```cpp
void consume() {
    m_availableItems.acquire();
    if (!queue_.empty()) {  // No synchronization, unsafe check
        int item = queue_.front();  
        queue_.pop();
        std::cout << "Consumed: " << item << "\n";
    }
    std::cout << "Entered Consume" << std::endl;
}
```

#### What It Does:
This function removes an item from the queue.

1. **`m_availableItems.acquire()`**: Decrements the semaphore. If the semaphore is 0, the consumer waits until an item is available.
2. **`if (!queue_.empty())`**: Checks if the queue is not empty. This is **unsafe** because multiple threads might access the queue simultaneously.
3. **`int item = queue_.front()`**: Retrieves the item at the front of the queue.
4. **`queue_.pop()`**: Removes the item from the queue.
5. **`std::cout << "Consumed: " << item << "\n"`**: Prints a message indicating that an item has been consumed.
6. **`std::cout << "Entered Consume" << std::endl`**: Prints a message indicating that the consumer has entered the function.

#### Why It’s Used:
The `consume` function simulates a consumer removing items from the queue. The semaphore ensures that consumers wait when the queue is empty.

---

### 5. **Producer Function**
```cpp
void producer(UnsafeQueue& queue, int numItems) {
    for (int i = 0; i < numItems; ++i) {
        queue.produce(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
```

#### What It Does:
This function simulates a producer generating items.

1. **`for (int i = 0; i < numItems; ++i)`**: A loop that runs `numItems` times. Each iteration produces one item.
2. **`queue.produce(i)`**: Calls the `produce` function to add the item `i` to the queue.
3. **`std::this_thread::sleep_for(std::chrono::milliseconds(10))`**: Pauses the thread for 10 milliseconds to simulate work.

#### Why It’s Used:
The `producer` function simulates a producer generating items and adding them to the queue. The delay makes the simulation more realistic.

---

### 6. **Consumer Function**
```cpp
void consumer(UnsafeQueue& queue, int numItems) {
    int consumed = 0;
    while (consumed < numItems) {
        queue.consume();
        ++consumed;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
```

#### What It Does:
This function simulates a consumer removing items.

1. **`int consumed = 0`**: Initializes a counter to track the number of items consumed.
2. **`while (consumed < numItems)`**: A loop that runs until the consumer has consumed `numItems` items.
3. **`queue.consume()`**: Calls the `consume` function to remove an item from the queue.
4. **`++consumed`**: Increments the counter.
5. **`std::this_thread::sleep_for(std::chrono::milliseconds(100))`**: Pauses the thread for 100 milliseconds to simulate work.

#### Why It’s Used:
The `consumer` function simulates a consumer removing items from the queue. The delay makes the simulation more realistic.

---

### 7. **Main Function**
```cpp
int main() {
    constexpr int numItems = 20;
    UnsafeQueue unsafeQueue;

    std::thread producerThread(producer, std::ref(unsafeQueue), numItems);
    std::thread consumerThread(consumer, std::ref(unsafeQueue), numItems);
    std::thread consumer2Thread(consumer, std::ref(unsafeQueue), numItems);

    producerThread.join();
    consumerThread.join();
    return 0;
}
```

#### What It Does:
This is the entry point of the program.

1. **`constexpr int numItems = 20`**: Defines the number of items to produce and consume.
2. **`UnsafeQueue unsafeQueue`**: Creates an instance of the `UnsafeQueue` class.
3. **`std::thread producerThread(producer, std::ref(unsafeQueue), numItems)`**: Creates a thread that runs the `producer` function.
4. **`std::thread consumerThread(consumer, std::ref(unsafeQueue), numItems)`**: Creates a thread that runs the `consumer` function.
5. **`std::thread consumer2Thread(consumer, std::ref(unsafeQueue), numItems)`**: Creates a second thread that runs the `consumer` function.
6. **`producerThread.join()`**: Waits for the producer thread to finish.
7. **`consumerThread.join()`**: Waits for the first consumer thread to finish.

#### Why It’s Used:
The `main` function sets up the producer and consumer threads and starts the simulation. The `join` calls ensure that the program waits for the threads to finish before exiting.

---

### Summary

This code demonstrates the **Producer-Consumer problem** in a multi-threaded environment. It highlights the importance of synchronization by intentionally leaving the queue **unsafe**, which can lead to data races. The next step would be to fix the code by adding proper synchronization mechanisms (e.g., mutexes) to make the queue thread-safe.