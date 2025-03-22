# Code Overview: main.cpp

This C++ code implements a classic **Producer-Consumer Problem**, which is a fundamental synchronization problem in concurrent programming. The purpose of this code is to demonstrate how to safely share a bounded buffer (a queue with a fixed size) between multiple threads, where one thread produces data (the producer) and another thread consumes data (the consumer). The code ensures that the producer and consumer can work together without causing race conditions, deadlocks, or buffer overflows.

### Problem Being Solved
The Producer-Consumer Problem arises when:
1. A **producer** generates data and adds it to a shared buffer (queue).
2. A **consumer** removes data from the buffer and processes it.
3. The buffer has a limited size, so the producer must wait if the buffer is full, and the consumer must wait if the buffer is empty.

The challenge is to ensure that:
- The producer and consumer do not access the buffer simultaneously (race conditions).
- The producer does not add data to a full buffer (overflow).
- The consumer does not try to remove data from an empty buffer (underflow).

### Approach Taken
The code uses **semaphores** and **mutexes** to solve this problem:
1. **Semaphores**:
   - `m_availableSlots`: Tracks the number of available slots in the buffer (initialized to the buffer size).
   - `m_availableItems`: Tracks the number of items in the buffer (initialized to 0).
2. **Mutex**:
   - `m_mutex`: Ensures exclusive access to the shared queue when adding or removing items.

The producer and consumer threads are synchronized using these mechanisms:
- The producer waits for an available slot (`m_availableSlots.acquire()`), adds an item to the queue, and signals that a new item is available (`m_availableItems.release()`).
- The consumer waits for an available item (`m_availableItems.acquire()`), removes an item from the queue, and signals that a slot is now available (`m_availableSlots.release()`).

### Overall Structure
The code is divided into three main parts:
1. **ProducerConsumerQueue Class**:
   - Manages the shared queue and synchronization mechanisms.
   - Provides thread-safe `produce()` and `consume()` methods.
2. **Producer and Consumer Functions**:
   - `producer()`: Generates items and adds them to the queue.
   - `consumer()`: Removes items from the queue and processes them.
3. **Main Function**:
   - Initializes the queue and starts the producer and consumer threads.
   - Waits for both threads to finish using `join()`.

### How the Parts Work Together
1. **Initialization**:
   - The `ProducerConsumerQueue` is created with a fixed size (`queueSize`).
   - The producer and consumer threads are started, sharing the same queue.
2. **Producer Thread**:
   - Generates items (integers from 0 to `numItems-1`) and adds them to the queue using `produce()`.
   - Waits for 50ms between producing items to simulate work.
3. **Consumer Thread**:
   - Removes items from the queue using `consume()` and processes them.
   - Waits for 100ms between consuming items to simulate work.
4. **Synchronization**:
   - The semaphores ensure that the producer waits if the queue is full and the consumer waits if the queue is empty.
   - The mutex ensures that only one thread accesses the queue at a time.

### Algorithms Used
1. **Semaphore-Based Synchronization**:
   - The producer and consumer use semaphores to coordinate access to the buffer.
   - This avoids busy-waiting and ensures efficient use of CPU resources.
2. **Mutex for Exclusive Access**:
   - The mutex ensures that the queue is accessed in a thread-safe manner.
3. **Bounded Buffer**:
   - The queue has a fixed size, and the semaphores enforce this limit.

### Key Concepts Demonstrated
- **Thread Safety**: The code ensures that shared resources (the queue) are accessed safely by multiple threads.
- **Synchronization**: Semaphores and mutexes are used to coordinate access between threads.
- **Concurrency**: The producer and consumer threads run concurrently, simulating real-world scenarios where multiple tasks happen simultaneously.

This code is an excellent example of how to handle shared resources in a multi-threaded environment while avoiding common pitfalls like race conditions and deadlocks.