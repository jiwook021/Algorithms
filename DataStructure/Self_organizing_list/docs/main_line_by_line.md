# Step-by-Step Explanation: main.cpp

Absolutely! Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into manageable sections, explain each part in simple terms, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the code’s design.

---

### **1. The `Strategy` Enum**
```cpp
enum class Strategy { MTF, Transpose, Count };
```

#### **What it does:**
- This defines a set of named constants (`MTF`, `Transpose`, `Count`) that represent the three strategies for reorganizing the list.

#### **Why it’s used:**
- Enums make the code more readable and maintainable. Instead of using magic numbers (e.g., `0`, `1`, `2`), we use meaningful names like `Strategy::MTF`.

#### **Technical terms:**
- **Enum (Enumeration):** A user-defined type that consists of a set of named constants.

---

### **2. The `SelfOrganizingList` Class**
```cpp
template<typename T>
class SelfOrganizingList {
    // ...
};
```

#### **What it does:**
- This is a **template class**, meaning it can work with any data type (`T`). For example, you can create a list of integers (`SelfOrganizingList<int>`) or strings (`SelfOrganizingList<std::string>`).

#### **Why it’s used:**
- Templates allow the class to be reusable for different data types without duplicating code.

#### **Technical terms:**
- **Template:** A feature in C++ that allows functions or classes to operate with generic types.

---

### **3. The `Node` Structure**
```cpp
struct Node {
    T data;
    int count;
    Node* next;
    Node(const T& item) : data(item), count(0), next(nullptr) {}
};
```

#### **What it does:**
- This defines the building block of the linked list. Each `Node` contains:
  - `data`: The actual item stored in the list.
  - `count`: The number of times the item has been accessed.
  - `next`: A pointer to the next node in the list.

#### **Why it’s used:**
- Linked lists are dynamic data structures that can grow or shrink as needed. Each node points to the next one, forming a chain.

#### **Technical terms:**
- **Linked List:** A linear data structure where each element (node) contains data and a pointer to the next node.
- **Pointer:** A variable that stores the memory address of another variable.

#### **Example:**
If the list contains `A -> B -> C`, the nodes would look like this:
```
Node 1: data = A, count = 0, next -> Node 2
Node 2: data = B, count = 0, next -> Node 3
Node 3: data = C, count = 0, next -> nullptr
```

---

### **4. The Constructor**
```cpp
SelfOrganizingList(Strategy s = Strategy::MTF) : head(nullptr), strategy(s) {}
```

#### **What it does:**
- Initializes a new `SelfOrganizingList` object.
- Sets the `head` pointer to `nullptr` (indicating an empty list).
- Sets the `strategy` to the provided value (defaulting to `Strategy::MTF`).

#### **Why it’s used:**
- Constructors ensure that objects are properly initialized when they are created.

---

### **5. The Destructor**
```cpp
~SelfOrganizingList() {
    while (head) {
        Node* temp = head;
        head = head->next;
        delete temp;
    }
}
```

#### **What it does:**
- Cleans up the list by deleting all nodes when the object is destroyed.
- It traverses the list, deleting each node one by one.

#### **Why it’s used:**
- Prevents memory leaks by freeing dynamically allocated memory.

#### **Technical terms:**
- **Memory Leak:** A situation where memory is allocated but not properly deallocated, leading to wasted resources.

---

### **6. The `insert` Function**
```cpp
void insert(const T& item) {
    Node* newNode = new Node(item);
    newNode->next = head;
    head = newNode;
}
```

#### **What it does:**
- Adds a new item to the front of the list.
- Creates a new node, sets its `next` pointer to the current `head`, and updates `head` to point to the new node.

#### **Why it’s used:**
- Inserting at the front is efficient (O(1) time complexity).

#### **Example:**
If the list is `B -> C` and we insert `A`, it becomes `A -> B -> C`.

---

### **7. The `find` Function**
This is the most complex part of the code, so let’s break it down step by step.

#### **Overview:**
- Searches for an item in the list.
- If found, increments its access count and reorganizes the list based on the selected strategy.

#### **Step-by-Step Breakdown:**

1. **Check if the list is empty:**
   ```cpp
   if (!head) return false;
   ```
   - If the list is empty (`head` is `nullptr`), return `false`.

2. **Handle the case where the item is at the head:**
   ```cpp
   if (head->data == item) {
       head->count++;
       return true;
   }
   ```
   - If the item is at the front, increment its count and return `true`.

3. **Traverse the list to find the item:**
   ```cpp
   Node* prev = head;
   Node* current = head->next;
   Node* prevPrev = nullptr;
   ```
   - Use three pointers: `prev` (previous node), `current` (current node), and `prevPrev` (node before `prev`).

4. **Search loop:**
   ```cpp
   while (current) {
       if (current->data == item) {
           current->count++;
           // Reorganization logic...
           return true;
       }
       prevPrev = prev;
       prev = current;
       current = current->next;
   }
   ```
   - Traverse the list until the item is found or the end is reached.

5. **Reorganization based on strategy:**
   - **Move-to-Front (MTF):**
     ```cpp
     prev->next = current->next;
     current->next = head;
     head = current;
     ```
     - Move the found item to the front.

   - **Transpose:**
     ```cpp
     prev->next = current->next;
     current->next = prev;
     if (prevPrev) {
         prevPrev->next = current;
     } else {
         head = current;
     }
     ```
     - Swap the found item with its predecessor.

   - **Count:**
     ```cpp
     Node* scan = head;
     Node* scanPrev = nullptr;
     while (scan != current && scan->count >= current->count) {
         scanPrev = scan;
         scan = scan->next;
     }
     if (scan != current) {
         prev->next = current->next;
         current->next = scan;
         if (scanPrev) {
             scanPrev->next = current;
         } else {
             head = current;
         }
     }
     ```
     - Move the found item to a position where all preceding nodes have higher or equal access counts.

---

### **8. The `display` Function**
```cpp
void display() const {
    Node* current = head;
    while (current) {
        std::cout << current->data;
        if (strategy == Strategy::Count) {
            std::cout << "(접근 횟수: " << current->count << ")";
        }
        std::cout << " -> ";
        current = current->next;
    }
    std::cout << "nullptr" << std::endl;
}
```

#### **What it does:**
- Traverses the list and prints each item.
- If the strategy is `Count`, it also prints the access count.

#### **Why it’s used:**
- Provides a way to visualize the list and verify its state.

---

### **9. The `main` Function**
- Demonstrates the functionality of the `SelfOrganizingList` class by testing all three strategies with a list of strings.

---

### **Summary**
This code implements a **Self-Organizing List** using a singly linked list. It dynamically reorganizes the list based on access patterns to improve search efficiency. The three strategies (MTF, Transpose, Count) offer different trade-offs between simplicity and performance. The code is modular, well-structured, and demonstrates key concepts like templates, linked lists, and dynamic memory management.

Let me know if you’d like further clarification on any part!