# Code Overview: main.cpp

### Purpose and Main Functionality of the Code

This C++ code demonstrates the use of the `std::map` container, which is a **sorted associative container** that stores key-value pairs. The purpose of the code is to showcase various operations that can be performed on a `std::map`, such as inserting, updating, erasing, and querying elements. The code also highlights how to use lambda functions for printing and handling the results of these operations.

#### Key Concepts and Functionality:
1. **`std::map` Container**:
   - A `std::map` is a container that stores elements as key-value pairs, where each key is unique.
   - The keys are automatically sorted in ascending order (by default), making it efficient for lookups and range-based operations.

2. **Lambda Functions**:
   - The code uses two lambda functions (`print_node` and `print_result`) to handle printing of key-value pairs and the results of insertion operations.
   - Lambda functions are anonymous functions that can be defined inline and are useful for short, reusable pieces of code.

3. **Operations Demonstrated**:
   - **Insertion**: Using `try_emplace`, `insert`, and the `[]` operator.
   - **Erasing**: Removing an element by its key.
   - **Querying**: Finding elements using `find` and checking the existence of a key using `count`.
   - **Iteration**: Looping through the map to print all key-value pairs.
   - **Clearing**: Removing all elements from the map.

4. **Problem Being Solved**:
   - The code does not solve a specific real-world problem but serves as an educational example to demonstrate how to work with `std::map` and its associated operations.

5. **Approach Taken**:
   - The code starts by defining two lambda functions for printing.
   - It then creates a `std::map` and performs various operations on it, printing the results at each step.
   - Finally, it iterates through the map to display its contents and clears it to show how the container behaves when empty.

6. **How the Parts Work Together**:
   - The lambda functions (`print_node` and `print_result`) are used throughout the code to provide consistent and reusable printing functionality.
   - The `main` function orchestrates the operations on the `std::map`, using the lambda functions to display the results.
   - The code progresses logically from insertion to querying, erasing, and finally clearing the map.

---

### Detailed Explanation of the Code

Let’s break down the code into its main components and explain each part in detail.

#### 1. **Lambda Functions**
```cpp
auto print_node = [](const auto &node) 
{
    std::cout << "[" << node.first << "] = " << node.second << '\n';
};
```
- **Purpose**: This lambda function prints a key-value pair in the format `[key] = value`.
- **Parameters**:
  - `const auto &node`: A reference to a key-value pair (e.g., a `std::pair` from the map).
- **Behavior**:
  - `node.first` accesses the key.
  - `node.second` accesses the value.
  - The function prints the key and value in a formatted way.

```cpp
auto print_result = [](auto const &pair) 
{
    std::cout << (pair.second ? "inserted: " : "ignored:  ");
    print_node(*pair.first);
};
```
- **Purpose**: This lambda function prints the result of an insertion operation.
- **Parameters**:
  - `auto const &pair`: A reference to the result of `try_emplace`, which is a `std::pair` containing an iterator and a boolean.
- **Behavior**:
  - `pair.second` is a boolean indicating whether the insertion was successful (`true`) or if the key already existed (`false`).
  - `pair.first` is an iterator pointing to the element in the map.
  - The function prints either "inserted" or "ignored" and then calls `print_node` to display the key-value pair.

#### 2. **Main Function**
```cpp
int main()
{
    using namespace std::literals;
    std::map<std::string, std::string> m;
```
- **Purpose**: The `main` function is the entry point of the program.
- **Details**:
  - `using namespace std::literals`: Enables the use of `"s"` suffix for `std::string` literals.
  - `std::map<std::string, std::string> m`: Creates an empty `std::map` where both keys and values are `std::string`.

#### 3. **Insertion Operations**
```cpp
print_result( m.try_emplace("a", "a"s) );
print_result( m.try_emplace("b", "abcd") );
print_result( m.try_emplace("c", 10, 'c') );
print_result( m.try_emplace("c", "Won't be inserted") );
```
- **Purpose**: Demonstrates the use of `try_emplace` to insert elements into the map.
- **Details**:
  - `try_emplace` inserts a key-value pair only if the key does not already exist.
  - The first three calls insert new keys ("a", "b", "c") with corresponding values.
  - The fourth call attempts to insert a value for the existing key "c", which is ignored.
  - The `print_result` lambda is used to display the outcome of each insertion.

#### 4. **Additional Insertion and Erasure**
```cpp
m.insert(std::make_pair("d", "4"));
m.insert(std::make_pair("e", "5"));
m["f"] = "6";
m["g"] = "7";
m.erase("d");
```
- **Purpose**: Shows alternative ways to insert elements and how to erase an element.
- **Details**:
  - `insert(std::make_pair(...))`: Inserts a key-value pair using `std::make_pair`.
  - `m["f"] = "6"`: Uses the `[]` operator to insert or update a value for a key.
  - `erase("d")`: Removes the key "d" from the map.

#### 5. **Querying the Map**
```cpp
std::cout << "Find b: " << m.find("b")->second << '\n';
std::cout << "Find f: " << m.find("f")->second << '\n';
std::cout << "a count: " << m.count("a") << '\n';
```
- **Purpose**: Demonstrates how to find elements and check for the existence of a key.
- **Details**:
  - `find("b")`: Returns an iterator to the element with key "b". The `->second` accesses its value.
  - `count("a")`: Returns `1` if the key exists, `0` otherwise.

#### 6. **Size and Iteration**
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
- **Purpose**: Checks if the map is empty, prints its size, and iterates through all elements.
- **Details**:
  - `empty()`: Returns `true` if the map is empty.
  - `size()`: Returns the number of elements in the map.
  - The `for` loop iterates through all key-value pairs and uses `print_node` to display them.

#### 7. **Clearing the Map**
```cpp
m.clear();
for (const auto &p : m) 
{ 
    print_node(p); 
}
```
- **Purpose**: Demonstrates how to clear the map and verify that it is empty.
- **Details**:
  - `clear()`: Removes all elements from the map.
  - The subsequent `for` loop shows that the map is now empty.

---

### Summary of the Code's Purpose
The code is a comprehensive demonstration of how to use `std::map` in C++. It covers:
- Insertion using `try_emplace`, `insert`, and the `[]` operator.
- Erasing elements.
- Querying elements using `find` and `count`.
- Iterating through the map.
- Clearing the map.

The use of lambda functions for printing makes the code modular and reusable. This example is ideal for learning how to work with associative containers in C++.