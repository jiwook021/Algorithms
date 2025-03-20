# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple language, examples, and diagrams to make everything clear, even for beginners.

---

### **1. Header Files and Includes**
```cpp
#include <iostream>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <stack>
#include <mutex>
#include <optional>
#include <variant>
#include <sstream>
```

#### **What It Does**
These are **header files** that provide functionality from the C++ Standard Library. For example:
- `<iostream>`: For input/output (e.g., `std::cout`).
- `<vector>`: For dynamic arrays (used for the stack).
- `<mutex>`: For thread synchronization (used to make memory operations thread-safe).

#### **Why It’s Used**
- These headers are included to provide the necessary tools for the VM to work.
- For example, `<vector>` is used to implement the stack, and `<mutex>` ensures thread safety.

---

### **2. Forward Declarations**
```cpp
class Memory;
class VirtualMachine;
```

#### **What It Does**
- These are **forward declarations** of the `Memory` and `VirtualMachine` classes.
- They tell the compiler that these classes exist, even though their full definitions come later.

#### **Why It’s Used**
- Forward declarations are used to avoid circular dependencies between classes.
- For example, if `Memory` needs to reference `VirtualMachine` and vice versa, forward declarations allow this.

---

### **3. Custom Exception Classes**
```cpp
class VMException : public std::runtime_error {
public:
    explicit VMException(const std::string& message) : std::runtime_error(message) {}
};

class StackOverflowException : public VMException {
public:
    explicit StackOverflowException() : VMException("Stack overflow") {}
};
```

#### **What It Does**
- These are **custom exception classes** that inherit from `std::runtime_error`.
- They are used to handle specific errors in the VM, such as stack overflow or invalid instructions.

#### **Why It’s Used**
- Custom exceptions make error handling more specific and meaningful.
- For example, `StackOverflowException` clearly indicates that the stack has exceeded its capacity.

#### **Example**
If the stack is full and you try to push another value, the VM throws a `StackOverflowException`.

---

### **4. Memory Class**
The `Memory` class manages the stack and heap.

#### **Value Type**
```cpp
using Value = std::variant<int, float, bool, void*>;
```

#### **What It Does**
- `std::variant` is a type-safe union that can hold one of several types (e.g., `int`, `float`, `bool`, or `void*`).
- This allows the stack to store different types of values.

#### **Why It’s Used**
- Using `std::variant` ensures type safety and avoids the need for unsafe type casting.

---

#### **Constructor**
```cpp
explicit Memory(size_t stackCapacity = 1024, size_t heapCapacity = 4096)
    : stackCapacity_(stackCapacity), heapCapacity_(heapCapacity) {
    heap_.resize(heapCapacity_, 0);
}
```

#### **What It Does**
- Initializes the stack and heap with default capacities (1024 for the stack, 4096 for the heap).
- Resizes the heap vector to the specified capacity and fills it with zeros.

#### **Why It’s Used**
- The constructor ensures that the memory is ready for use when the VM starts.

---

#### **Stack Operations**
```cpp
void pushStack(const Value& value) {
    std::lock_guard<std::mutex> lock(stackMutex_);
    
    if (stack_.size() >= stackCapacity_) {
        throw StackOverflowException();
    }
    
    stack_.push_back(value);
}
```

#### **What It Does**
- Pushes a value onto the stack.
- Uses a mutex (`stackMutex_`) to ensure thread safety.
- Throws a `StackOverflowException` if the stack is full.

#### **Why It’s Used**
- The stack is used for temporary storage during program execution.
- Mutexes ensure that multiple threads can’t modify the stack at the same time.

#### **Example**
If the stack capacity is 1024 and you try to push a 1025th value, the VM throws a `StackOverflowException`.

---

#### **Heap Operations**
```cpp
size_t allocateHeap(size_t size) {
    std::lock_guard<std::mutex> lock(heapMutex_);
    
    if (heapAllocPtr_ + size > heapCapacity_) {
        throw HeapAllocationException();
    }
    
    size_t ptr = heapAllocPtr_;
    heapAllocPtr_ += size;
    return ptr;
}
```

#### **What It Does**
- Allocates a block of memory on the heap.
- Uses a **bump allocator** (a simple memory allocation strategy).
- Throws a `HeapAllocationException` if there’s not enough memory.

#### **Why It’s Used**
- The heap is used for dynamic memory allocation (e.g., for arrays or objects).
- A bump allocator is simple and efficient for demonstration purposes.

#### **Example**
If the heap capacity is 4096 and you try to allocate 4097 bytes, the VM throws a `HeapAllocationException`.

---

### **5. Main Function**
```cpp
int main() {
    try {
        VirtualMachine vm;
        
        // Test 1: Factorial program
        std::cout << "\n========== Test 1: Factorial of 5 ==========\n";
        auto factorialProgram = TestPrograms::createFactorialProgram();
        vm.loadProgram(factorialProgram);
        vm.run();
    } catch (const VMException& e) {
        std::cerr << "VM Error: " << e.what() << std::endl;
    }
}
```

#### **What It Does**
- Creates a `VirtualMachine` instance.
- Loads and runs a factorial program.
- Catches and handles any VM exceptions.

#### **Why It’s Used**
- The `main` function is the entry point of the program.
- It demonstrates how the VM works by running test programs.

#### **Example**
If the factorial program tries to divide by zero, the VM throws a `DivisionByZeroException`, which is caught and handled in the `main` function.

---

### **6. Thread Safety**
```cpp
std::lock_guard<std::mutex> lock(stackMutex_);
```

#### **What It Does**
- Locks the mutex to ensure that only one thread can access the stack at a time.
- Automatically unlocks the mutex when the `lock_guard` goes out of scope.

#### **Why It’s Used**
- Mutexes prevent race conditions (e.g., two threads trying to modify the stack simultaneously).

#### **Example**
If two threads try to push values onto the stack at the same time, the mutex ensures that one thread waits for the other to finish.

---

### **7. Diagrams**

#### **Stack and Heap Memory**
```
+-------------------+       +-------------------+
|      Stack        |       |       Heap        |
+-------------------+       +-------------------+
| Value 1 (int)     |       | Byte 0: 0x00      |
| Value 2 (float)   |       | Byte 1: 0x01      |
| ...               |       | ...               |
+-------------------+       +-------------------+
```

- The **stack** stores temporary values (e.g., function arguments, local variables).
- The **heap** stores dynamically allocated memory (e.g., arrays, objects).

---

### **Summary**
This code implements a **stack-based virtual machine** with:
- A **memory model** (stack and heap).
- **Thread-safe operations** using mutexes.
- **Custom exceptions** for error handling.
- A **main function** to test the VM.

Each part of the code is designed to be **modular**, **thread-safe**, and **easy to extend**. By breaking it down step by step, we’ve made it accessible to everyone, from beginners to experts!