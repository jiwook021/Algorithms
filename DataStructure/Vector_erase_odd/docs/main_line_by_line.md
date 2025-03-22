# Step-by-Step Explanation: main.cpp

Let’s break down the code **line by line** and explain it in **extreme detail**, as if teaching someone who is just learning to program. I’ll explain every concept, term, and operation, and provide examples and diagrams where necessary.

---

### **1. Header Files**
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
```

#### What it does:
- These lines include necessary libraries for the program to work:
  - `<iostream>`: Provides input/output functionality (e.g., printing to the console).
  - `<vector>`: Provides the `std::vector` container, which is a dynamic array that can grow or shrink in size.
  - `<algorithm>`: Provides useful algorithms like `std::remove` and `std::remove_if`.

#### Why it’s used:
- Without these libraries, the program wouldn’t have access to tools like `std::vector`, `std::cout`, or algorithms like `std::remove`.

---

### **2. Main Function**
```cpp
int main()
{
```
#### What it does:
- This is the entry point of the program. When the program runs, execution starts here.

#### Why it’s used:
- Every C++ program must have a `main` function. It’s where the program begins and ends.

---

### **3. Initialize the Vector**
```cpp
std::vector<int> v {1, 2, 3, 2, 5, 2, 6, 2, 4, 8};
```

#### What it does:
- Creates a `std::vector<int>` named `v` and initializes it with the values `{1, 2, 3, 2, 5, 2, 6, 2, 4, 8}`.

#### What is a `std::vector`?
- A `std::vector` is a dynamic array. Unlike a regular array, it can grow or shrink in size automatically.
- Example: If you add or remove elements, the vector adjusts its size.

#### Why it’s used:
- Vectors are flexible and easy to use for storing collections of data.

#### Visual Representation of the Vector:
```
Index: 0 1 2 3 4 5 6 7 8 9
Value:1 2 3 2 5 2 6 2 4 8
```

---

### **4. Remove All Occurrences of `2`**
```cpp
{
    const auto new_end (remove(begin(v), end(v), 2));
    v.erase(new_end, end(v));
}
```

#### What it does:
- Removes all occurrences of the number `2` from the vector.

#### Step-by-Step Breakdown:
1. **`std::remove`**:
   - This algorithm moves all `2`s to the end of the vector.
   - It doesn’t actually delete the elements; it just rearranges them.
   - It returns an iterator (`new_end`) pointing to the new logical end of the vector.

   Example:
   - Before `std::remove`:
     ```
     Vector: 1 2 3 2 5 2 6 2 4 8
     ```
   - After `std::remove`:
     ```
     Vector: 1 3 5 6 4 8 2 2 2 2
     ```
     - The `2`s are moved to the end.
     - `new_end` points to the first `2` at the end.

2. **`v.erase(new_end, end(v))`**:
   - This actually deletes the elements from `new_end` to the end of the vector.
   - After this, the vector looks like:
     ```
     Vector: 1 3 5 6 4 8
     ```

#### Why this approach is used:
- `std::remove` is efficient because it doesn’t actually delete elements (which can be slow). Instead, it rearranges them, and `erase` removes them in one go.

---

### **5. Print the Vector After Removing `2`s**
```cpp
std::cout << "Size of vector: " << v.size();
std::cout << "\nInput Vectors\n";
for (auto i : v) {
    std::cout << i << " ";
}
```

#### What it does:
- Prints the size of the vector and its contents.

#### Step-by-Step Breakdown:
1. **`v.size()`**:
   - Returns the number of elements in the vector.
   - After removing `2`s, the size is `6`.

2. **Range-based `for` loop**:
   - Iterates over each element in the vector and prints it.
   - `auto i : v` means "for each element `i` in `v`".

#### Why it’s used:
- To show the intermediate result after removing `2`s.

---

### **6. Remove All Odd Numbers**
```cpp
{
    const auto odd ([](int i) { return i % 2 != 0; });
    const auto new_end (remove_if(begin(v), end(v), odd));
    v.erase(new_end, end(v));
}
```

#### What it does:
- Removes all odd numbers from the vector.

#### Step-by-Step Breakdown:
1. **Lambda Function**:
   - `[](int i) { return i % 2 != 0; }` is a lambda function (a small, anonymous function).
   - It checks if a number is odd (`i % 2 != 0`).

2. **`std::remove_if`**:
   - Similar to `std::remove`, but it uses the lambda function to decide which elements to move.
   - Moves all odd numbers to the end of the vector.
   - Returns an iterator (`new_end`) pointing to the new logical end.

   Example:
   - Before `std::remove_if`:
     ```
     Vector: 1 3 5 6 4 8
     ```
   - After `std::remove_if`:
     ```
     Vector: 6 4 8 1 3 5
     ```
     - The odd numbers (`1`, `3`, `5`) are moved to the end.
     - `new_end` points to the first odd number at the end.

3. **`v.erase(new_end, end(v))`**:
   - Deletes the odd numbers from the vector.
   - After this, the vector looks like:
     ```
     Vector: 6 4 8
     ```

#### Why this approach is used:
- `std::remove_if` is flexible because it allows custom conditions (like checking for odd numbers).

---

### **7. Optimize Memory Usage**
```cpp
v.shrink_to_fit();
```

#### What it does:
- Reduces the vector’s capacity to match its size, freeing up unused memory.

#### Why it’s used:
- After removing elements, the vector might still have extra memory allocated. `shrink_to_fit` ensures the vector uses only as much memory as needed.

---

### **8. Print the Final Result**
```cpp
std::cout << "New Size: " << v.size() << std::endl;
for (auto i : v) {
    std::cout << i << " ";
}
std::cout << '\n';
```

#### What it does:
- Prints the final size of the vector and its contents.

#### Why it’s used:
- To show the final result after removing odd numbers.

---

### **Final Output**
The program outputs:
```
Size of vector: 6
Input Vectors
1 3 5 6 4 8 
Erase the odd numbers
New Size: 3
6 4 8 
```

---

### **Summary of the Code’s Flow**
1. Initialize a vector with some values.
2. Remove all `2`s and print the result.
3. Remove all odd numbers and print the final result.
4. Optimize memory usage with `shrink_to_fit`.

This code is a great example of how to manipulate vectors in C++ using Standard Library tools. It’s efficient, easy to read, and demonstrates key programming concepts like loops, conditionals, and algorithms.