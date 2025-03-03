# When should you use `shared_ptr`, `unique_ptr`, and `weak_ptr` in C++?

In C++, smart pointers are used to manage the lifecycle of objects that are dynamically allocated (using `new`). They help to ensure that memory is properly deallocated, thus helping to prevent memory leaks and dangling pointers. The three primary types of smart pointers in the C++ Standard Library are `std::shared_ptr`, `std::unique_ptr`, and `std::weak_ptr`. Each serves a specific purpose and is used in different scenarios:

### 1. `std::unique_ptr`
`std::unique_ptr` is used when you have a single owner for a dynamically allocated object. It provides exclusive ownership, meaning no two `unique_ptr` instances can manage the same object. Here are some specific scenarios where `unique_ptr` is appropriate:
- To manage resources in a class where the resource should not be shared or copied, such as file handles or mutexes.
- When returning objects from a function to avoid the overhead of copying and to use move semantics.
- In containers that require unique ownership semantics.

**Key Benefits:**
- Lightweight and efficient, with minimal overhead compared to raw pointers.
- Automatically deletes the object it manages when the `unique_ptr` goes out of scope.

**Example Usage:**
```cpp
std::unique_ptr<int> p = std::make_unique<int>(10);
```

### 2. `std::shared_ptr`
`std::shared_ptr` is used when you want to share ownership of a dynamically allocated object between multiple pointers. The object will only be destroyed when the last `shared_ptr` managing it is destroyed or reset.

**Key Scenarios:**
- When multiple parts of your program must access the same object and it is not clear when the object should be deleted.
- In implementing complex data structures like graphs or trees where multiple elements can own the same node.

**Key Benefits:**
- Manages the underlying object via reference counting, automatically deleting the managed object when the reference count reaches zero.
- Can be safely used in standard containers.

**Example Usage:**
```cpp
std::shared_ptr<int> p1 = std::make_shared<int>(10);
std::shared_ptr<int> p2 = p1; // Both now own the memory.
```

### 3. `std::weak_ptr`
`std::weak_ptr` is used in conjunction with `std::shared_ptr` to break cycles which can lead to memory leaks. `weak_ptr` holds a non-owning (weak) reference to an object that is managed by `shared_ptr`.

**Key Scenarios:**
- To keep a reference to an object without extending its lifetime, such as in caching scenarios where objects can expire independently.
- To break circular references in data structures managed by `shared_ptr`. For example, in a parent-child relationship where both parent and child have `shared_ptr` to each other.

**Key Benefits:**
- Does not affect the lifetime of the object it references.
- Can be converted to a `shared_ptr` to temporarily obtain ownership.

**Example Usage:**
```cpp
std::shared_ptr<int> sp = std::make_shared<int>(10);
std::weak_ptr<int> wp = sp;
// Use wp by converting to shared_ptr
std::shared_ptr<int> sp2 = wp.lock();
if (sp2) {
    // Use sp2
}
```

### Conclusion
Choose `unique_ptr` when you need unique ownership, `shared_ptr` when you need shared ownership, and `weak_ptr` to avoid ownership but still refer to an object managed by `shared_ptr`. Correct usage of these smart pointers aids in writing robust and memory-safe C++ code.