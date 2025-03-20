# Implement a thread-safe queue using `std::mutex` or `std::condition_variable`.

To implement a thread-safe queue in C++ using `std::mutex` and `std::condition_variable`, we need to ensure that multiple threads can simultaneously push items to the queue and pop items from it without causing data races or deadlocks. The `std::mutex` will be used to synchronize access to the queue, while the `std::condition_variable` will be used to wait for a condition (like the queue being non-empty) to become true.

Below is a simple implementation of a thread-safe queue template class using these mechanisms. This class supports basic operations such as `push` (to add items to the queue) and `pop` (to remove items from the queue). It also uses condition variables to efficiently handle situations where a thread tries to pop an item from an empty queue.

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <exception>

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;

public:
    ThreadSafeQueue() = default;

    // Delete copy constructor and assignment operator to prevent copying
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // Push an item to the queue.
    void push(T value) {
        std::lock_guard<std::mutex> lock(mtx); // Lock the mutex until the end of the scope
        queue.push(std::move(value));
        cv.notify_one(); // Notify one waiting thread
    }

    // Try to pop an item from the queue. Returns true if successful, false if the queue is empty.
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mtx); // Lock the mutex until the end of the scope
        if (queue.empty()) {
            return false;
        }
        value = std::move(queue.front());
        queue.pop();
        return true;
    }

    // Pop an item from the queue. Blocks if the queue is empty until an item is available.
    void wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mtx); // This lock can be unlocked and relocked
        cv.wait(lock, [this]{ return !queue.empty(); }); // Wait until the queue is not empty
        value = std::move(queue.front());
        queue.pop();
    }

    // Check if the queue is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }

    // Get the size of the queue
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }
};
```

### Explanation
1. **Mutex (`mtx`)**: This mutex is used to synchronize access to the queue. It ensures that only one thread can modify the queue at any given time.

2. **Condition Variable (`cv`)**: This condition variable is used in conjunction with the mutex to wait for a specific condition (non-empty queue) to be true before continuing execution.

3. **Member Functions**:
   - **`push`**: Locks the mutex, adds an item to the queue, and then notifies one of the waiting threads that an item is available.
   - **`try_pop`**: Tries to pop an item from the queue without blocking. It returns false if the queue is empty.
   - **`wait_and_pop`**: Waits (if necessary) until the queue is not empty and then pops an item from the queue. This function blocks if the queue is empty.
   - **`empty` and `size`**: Return the emptiness and size of the queue, respectively, with mutex protection to ensure thread safety.

This implementation ensures that the queue operations are thread-safe and that threads efficiently wait for conditions (like the queue having items) rather than polling or busy-waiting.