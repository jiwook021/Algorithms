# Code Overview: main.cpp

### Purpose of the Code

This C++ code is designed to demonstrate and compare how different container types (`std::vector` and `std::array`) handle **out-of-bounds access** in C++. Specifically, it shows the difference between using the **subscript operator (`[]`)** and the **`at()` method** when accessing elements beyond the container's allocated size.

The code highlights two key concepts:
1. **Unsafe Access (`[]` operator):** Accessing elements using the `[]` operator does not perform bounds checking, which can lead to undefined behavior (e.g., reading garbage values or crashing the program).
2. **Safe Access (`at()` method):** The `at()` method performs bounds checking and throws an exception (`std::out_of_range`) if the index is out of bounds, allowing the program to handle the error gracefully.

The code also demonstrates how to use **exception handling** (`try-catch`) to manage runtime errors caused by invalid memory access.

---

### Main Functionality and Algorithms

1. **Container Initialization:**
   - The code creates two containers:
     - A `std::vector<int>` of size `1000`.
     - A `std::array<int, 1000>` of fixed size `1000`.
   - Both containers are filled with **ascending integers** using the `std::iota` algorithm, which assigns sequential values starting from `0`.

2. **Out-of-Bounds Access:**
   - The code intentionally accesses elements **beyond the container's size**:
     - Using the `[]` operator: This does not perform bounds checking and may result in undefined behavior.
     - Using the `at()` method: This performs bounds checking and throws an exception if the index is invalid.

3. **Exception Handling:**
   - The code uses a `try-catch` block to catch the `std::out_of_range` exception thrown by the `at()` method. This allows the program to handle the error gracefully by printing an error message.

4. **Comparison of `std::vector` and `std::array`:**
   - The code demonstrates that the behavior of `std::vector` and `std::array` is identical when it comes to out-of-bounds access. Both containers throw exceptions when using `at()` and exhibit undefined behavior when using `[]`.

---

### Overall Structure

The code is structured as follows:
1. **Header Includes:**
   - The necessary headers (`<iostream>`, `<vector>`, `<array>`, and `<numeric>`) are included to enable the use of standard library features.

2. **Main Function:**
   - The `main()` function is the entry point of the program.
   - A constant `container_size` is defined to specify the size of the containers.

3. **`std::vector` Section:**
   - A `std::vector<int>` is created and filled with ascending integers using `std::iota`.
   - Out-of-bounds access is demonstrated using both the `[]` operator and the `at()` method.
   - The `try-catch` block handles the exception thrown by `at()`.

4. **`std::array` Section:**
   - A `std::array<int, container_size>` is created and filled with ascending integers using `std::iota`.
   - Out-of-bounds access is demonstrated using both the `[]` operator and the `at()` method.
   - The `try-catch` block handles the exception thrown by `at()`.

5. **Output:**
   - The program prints the results of out-of-bounds access attempts, including any error messages from caught exceptions.

---

### Problem Being Solved

The code addresses the following problem:
- **How do different container types (`std::vector` and `std::array`) handle out-of-bounds access in C++?**
- **What is the difference between using the `[]` operator and the `at()` method for accessing elements?**

By demonstrating the behavior of both containers and access methods, the code educates the user on the importance of bounds checking and exception handling in C++.

---

### Approach Taken

The code takes a **practical, hands-on approach** to demonstrate the concepts:
1. **Initialization:**
   - Both containers are initialized with a fixed size and filled with sequential values to provide a predictable starting point.

2. **Out-of-Bounds Access:**
   - The code intentionally accesses elements beyond the container's size to trigger undefined behavior or exceptions.

3. **Exception Handling:**
   - The `try-catch` block is used to catch and handle exceptions, showing how to gracefully manage runtime errors.

4. **Comparison:**
   - The behavior of `std::vector` and `std::array` is compared to show that they behave similarly in terms of bounds checking.

---

### How the Parts Work Together

1. **Initialization and Filling:**
   - The `std::iota` algorithm fills both containers with sequential values, ensuring they are in a known state before testing out-of-bounds access.

2. **Out-of-Bounds Access:**
   - The `[]` operator is used first to demonstrate unsafe access, which may result in undefined behavior.
   - The `at()` method is used next to demonstrate safe access, which throws an exception if the index is invalid.

3. **Exception Handling:**
   - The `try-catch` block catches the `std::out_of_range` exception and prints an error message, showing how to handle such errors gracefully.

4. **Output:**
   - The program prints the results of each access attempt, allowing the user to observe the differences in behavior between the `[]` operator and the `at()` method.

---

### Summary

This code serves as an educational tool to demonstrate:
- The dangers of out-of-bounds access in C++.
- The difference between unsafe (`[]`) and safe (`at()`) access methods.
- How to use exception handling to manage runtime errors.
- The similarities between `std::vector` and `std::array` in terms of bounds checking.

By running this code, a user can observe firsthand the consequences of out-of-bounds access and learn how to write safer, more robust C++ programs.