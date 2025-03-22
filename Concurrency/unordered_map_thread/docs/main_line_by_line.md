# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple language, examples, and diagrams to make everything clear, even for beginners.

---

### **1. Header Files**
```cpp
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <optional>
#include <stdexcept>
#include <vector>
#include <ctime>
#include <chrono>
#include <sstream>
#include <string>
#include <iomanip>
#include <random>
```

#### **What It Does**
These are **header files** that provide functionality for:
- Input/output (`<iostream>`)
- Smart pointers (`<memory>`)
- Thread synchronization (`<mutex>` and `<thread>`)
- Optional values (`<optional>`)
- Error handling (`<stdexcept>`)
- Dynamic arrays (`<vector>`)
- Time manipulation (`<ctime>` and `<chrono>`)
- String manipulation (`<sstream>` and `<string>`)
- Formatting (`<iomanip>`)
- Random number generation (`<random>`)

#### **Why It’s Used**
These headers are necessary for the program to use advanced C++ features like threading, memory management, and time-based operations.

---

### **2. Class Definition**
```cpp
template <typename T>
class ThreadSafeLinkedList {
    // Class implementation
};
```

#### **What It Does**
This defines a **template class** called `ThreadSafeLinkedList`. A template allows the class to work with any data type (`T`), such as `int`, `string`, or custom objects.

#### **Why It’s Used**
Templates make the class reusable for different data types without rewriting the code.

---

### **3. Node Structure**
```cpp
struct Node {
    T data;
    std::shared_ptr<Node> next;
    mutable std::mutex mutex;

    explicit Node(const T& value) : data(value), next(nullptr) {}
};
```

#### **What It Does**
This defines a **node** in the linked list. Each node contains:
- `data`: The value stored in the node.
- `next`: A pointer to the next node in the list.
- `mutex`: A lock to ensure thread-safe access to the node.

#### **Why It’s Used**
- `std::shared_ptr<Node>` ensures that the node is automatically deallocated when no longer needed.
- `mutable std::mutex` allows the mutex to be modified even in `const` functions, which is necessary for thread safety.

#### **Example**
If the list stores integers, a node might look like this:
```
Node 1: data = 5, next -> Node 2
Node 2: data = 10, next -> nullptr
```

---

### **4. Member Variables**
```cpp
std::shared_ptr<Node> head;
mutable std::mutex head_mutex;
std::size_t size_;
mutable std::mutex size_mutex;
mutable std::mutex cout_mutex;
```

#### **What It Does**
- `head`: A pointer to the first node in the list.
- `head_mutex`: A mutex to protect the `head` pointer.
- `size_`: The number of nodes in the list.
- `size_mutex`: A mutex to protect the `size_` variable.
- `cout_mutex`: A mutex to protect console output (used for debugging).

#### **Why It’s Used**
- Mutexes ensure that only one thread can modify `head` or `size_` at a time, preventing race conditions.

---

### **5. Constructors and Assignment Operators**
#### **Default Constructor**
```cpp
ThreadSafeLinkedList() : head(nullptr), size_(0) {}
```
- Initializes an empty list with `head` pointing to `nullptr` and `size_` set to 0.

#### **Deleted Copy Constructor and Assignment Operator**
```cpp
ThreadSafeLinkedList(const ThreadSafeLinkedList&) = delete;
ThreadSafeLinkedList& operator=(const ThreadSafeLinkedList&) = delete;
```
- Prevents copying of the list, as copying a thread-safe list could lead to race conditions.

#### **Move Constructor**
```cpp
ThreadSafeLinkedList(ThreadSafeLinkedList&& other) noexcept {
    std::lock_guard<std::mutex> lock_head(other.head_mutex);
    std::lock_guard<std::mutex> lock_size(other.size_mutex);
    
    head = std::move(other.head);
    size_ = other.size_;
    other.size_ = 0;
}
```
- Transfers ownership of the list from `other` to `this`.
- Locks `other`'s mutexes to ensure thread safety during the transfer.

#### **Move Assignment Operator**
```cpp
ThreadSafeLinkedList& operator=(ThreadSafeLinkedList&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(head_mutex, size_mutex, other.head_mutex, other.size_mutex);
        
        head = std::move(other.head);
        size_ = other.size_;
        other.size_ = 0;
    }
    return *this;
}
```
- Similar to the move constructor but handles assignment.

#### **Why It’s Used**
- Move semantics allow efficient transfer of resources (e.g., nodes) between objects without copying.

---

### **6. `push_front` Function**
```cpp
void push_front(const T& value) {
    auto new_node = std::make_shared<Node>(value);
    
    std::lock_guard<std::mutex> lock(head_mutex);
    
    new_node->next = head;
    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Deliberate delay
    head = new_node;
}
```

#### **What It Does**
- Inserts a new node at the beginning of the list.
- Locks `head_mutex` to ensure thread safety.
- Adds a deliberate delay to simulate real-world concurrency scenarios.

#### **Why It’s Used**
- Demonstrates how to safely modify shared data in a multi-threaded environment.

#### **Example**
If the list is `[2, 3]` and we insert `1`, it becomes `[1, 2, 3]`.

---

### **7. `main` Function**
```cpp
int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    ThreadSafeLinkedList<int> list;
    
    list.push_front(3);
    list.push_front(2);
    list.push_front(1);
    
    std::cout << "Initial List: " << list.to_string() << std::endl << std::endl;
    
    const int num_threads = 10;
    const int operations_per_thread = 10;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(test_thread<int>, std::ref(list), i, operations_per_thread);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "Final List: " << list.to_string() << std::endl;
    std::cout << "List size: " << list.size() << std::endl;
    
    return 0;
}
```

#### **What It Does**
1. Initializes a random number generator.
2. Creates a `ThreadSafeLinkedList` of integers.
3. Inserts initial values (`1`, `2`, `3`).
4. Spawns 10 threads, each performing 10 operations on the list.
5. Waits for all threads to finish.
6. Prints the final state of the list.

#### **Why It’s Used**
- Demonstrates how the list behaves under concurrent access.

#### **Diagram**
```
Thread 1: Insert 4 -> [4, 1, 2, 3]
Thread 2: Insert 5 -> [5, 4, 1, 2, 3]
...
Final List: [10, 9, 8, ..., 1]
```

---

### **Summary**
This code is a **thread-safe singly linked list** that uses fine-grained locking to allow concurrent operations. It demonstrates modern C++ features like smart pointers, mutexes, and move semantics. The `main` function tests the list’s thread safety by performing concurrent insertions. By breaking down each component, we’ve made the code accessible to learners of all levels!