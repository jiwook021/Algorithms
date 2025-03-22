# Step-by-Step Explanation: main.cpp

### Comprehensive Step-by-Step Explanation of the Code

Let’s go through the code line by line, breaking it down into digestible parts. I’ll explain each section in simple terms, define technical concepts, and provide examples where necessary. By the end, you’ll have a deep understanding of how this code works.

---

### 1. **Header Files**
```cpp
#include <iostream>
#include <utility>
#include <string>
#include <map>
```
- **What it does**: These lines include necessary libraries for the program.
  - `<iostream>`: Provides input/output functionality (e.g., `std::cout` for printing to the console).
  - `<utility>`: Provides utilities like `std::pair`, which is used to store key-value pairs.
  - `<string>`: Provides the `std::string` class for working with text.
  - `<map>`: Provides the `std::map` container, which stores key-value pairs in a sorted order.
- **Why it’s used**: These libraries are essential for the program to perform tasks like printing, storing strings, and using the `std::map` container.

---

### 2. **Lambda Functions**
#### a. `print_node`
```cpp
auto print_node = [](const auto &node) 
{
    std::cout << "[" << node.first << "] = " << node.second << '\n';
};
```
- **What it does**: This is a **lambda function** (an anonymous function) that prints a key-value pair in the format `[key] = value`.
- **Breakdown**:
  - `auto print_node = []`: Defines a lambda function named `print_node`.
  - `(const auto &node)`: The function takes a single parameter, `node`, which is a reference to a key-value pair.
  - `node.first`: Accesses the key.
  - `node.second`: Accesses the value.
  - `std::cout << ...`: Prints the key and value in a formatted way.
- **Example**:
  - If `node` is `{"a", "apple"}`, the function prints `[a] = apple`.
- **Why it’s used**: This function is reusable and makes it easy to print key-value pairs consistently throughout the code.

#### b. `print_result`
```cpp
auto print_result = [](auto const &pair) 
{
    std::cout << (pair.second ? "inserted: " : "ignored:  ");
    print_node(*pair.first);
};
```
- **What it does**: This lambda function prints the result of an insertion operation.
- **Breakdown**:
  - `auto print_result = []`: Defines a lambda function named `print_result`.
  - `(auto const &pair)`: The function takes a single parameter, `pair`, which is the result of `try_emplace`.
  - `pair.second`: A boolean indicating whether the insertion was successful (`true`) or if the key already existed (`false`).
  - `pair.first`: An iterator pointing to the element in the map.
  - `print_node(*pair.first)`: Calls `print_node` to print the key-value pair.
- **Example**:
  - If `pair.second` is `true`, it prints `inserted: [key] = value`.
  - If `pair.second` is `false`, it prints `ignored: [key] = value`.
- **Why it’s used**: This function provides a clear way to display the outcome of insertion operations.

---

### 3. **Main Function**
```cpp
int main()
{
    using namespace std::literals;
    std::map<std::string, std::string> m;
```
- **What it does**: The `main` function is the entry point of the program. It initializes a `std::map` to store key-value pairs.
- **Breakdown**:
  - `using namespace std::literals`: Allows the use of `"s"` suffix for `std::string` literals (e.g., `"a"s` creates a `std::string`).
  - `std::map<std::string, std::string> m`: Creates an empty `std::map` where both keys and values are `std::string`.
- **Why it’s used**: The `main` function orchestrates all the operations on the map.

---

### 4. **Insertion Operations**
```cpp
print_result( m.try_emplace("a", "a"s) );
print_result( m.try_emplace("b", "abcd") );
print_result( m.try_emplace("c", 10, 'c') );
print_result( m.try_emplace("c", "Won't be inserted") );
```
- **What it does**: Inserts key-value pairs into the map using `try_emplace`.
- **Breakdown**:
  - `try_emplace(key, value)`: Inserts a key-value pair only if the key does not already exist.
  - The first three calls insert new keys ("a", "b", "c") with corresponding values.
  - The fourth call attempts to insert a value for the existing key "c", which is ignored.
  - `print_result` is used to display the outcome of each insertion.
- **Example**:
  - `m.try_emplace("a", "a"s)` inserts `{"a", "a"}` and prints `inserted: [a] = a`.
  - `m.try_emplace("c", "Won't be inserted")` prints `ignored: [c] = cccccc` (since "c" already exists).
- **Why it’s used**: `try_emplace` is efficient and avoids unnecessary copies.

---

### 5. **Additional Insertion and Erasure**
```cpp
m.insert(std::make_pair("d", "4"));
m.insert(std::make_pair("e", "5"));
m["f"] = "6";
m["g"] = "7";
m.erase("d");
```
- **What it does**: Shows alternative ways to insert elements and how to erase an element.
- **Breakdown**:
  - `insert(std::make_pair(...))`: Inserts a key-value pair using `std::make_pair`.
  - `m["f"] = "6"`: Uses the `[]` operator to insert or update a value for a key.
  - `erase("d")`: Removes the key "d" from the map.
- **Why it’s used**: Demonstrates different methods for modifying the map.

---

### 6. **Querying the Map**
```cpp
std::cout << "Find b: " << m.find("b")->second << '\n';
std::cout << "Find f: " << m.find("f")->second << '\n';
std::cout << "a count: " << m.count("a") << '\n';
```
- **What it does**: Finds elements and checks for the existence of a key.
- **Breakdown**:
  - `find("b")`: Returns an iterator to the element with key "b". The `->second` accesses its value.
  - `count("a")`: Returns `1` if the key exists, `0` otherwise.
- **Why it’s used**: Shows how to retrieve and verify elements in the map.

---

### 7. **Size and Iteration**
```cpp
if(!m.empty()) 
{
    std::cout << "m size: " << m.size() << '\n';
}

for (const auto &p : m) 
{ 
    print_node(p); 
}
```
- **What it does**: Checks if the map is empty, prints its size, and iterates through all elements.
- **Breakdown**:
  - `empty()`: Returns `true` if the map is empty.
  - `size()`: Returns the number of elements in the map.
  - The `for` loop iterates through all key-value pairs and uses `print_node` to display them.
- **Why it’s used**: Demonstrates how to inspect and traverse the map.

---

### 8. **Clearing the Map**
```cpp
m.clear();
for (const auto &p : m) 
{ 
    print_node(p); 
}
```
- **What it does**: Clears the map and verifies that it is empty.
- **Breakdown**:
  - `clear()`: Removes all elements from the map.
  - The subsequent `for` loop shows that the map is now empty.
- **Why it’s used**: Shows how to reset the map.

---

### Summary
This code is a hands-on tutorial for working with `std::map` in C++. It covers:
- Insertion (`try_emplace`, `insert`, `[]` operator).
- Erasing elements.
- Querying elements (`find`, `count`).
- Iterating through the map.
- Clearing the map.

The use of lambda functions for printing makes the code modular and reusable. This example is ideal for learning how to work with associative containers in C++.