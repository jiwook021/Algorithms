# Code Overview: main.cpp

### Purpose and Main Functionality of the Code

This C++ program demonstrates how to manipulate a `std::vector<int>` by removing specific elements based on certain conditions. The code performs two main operations:

1. **Removing all occurrences of a specific value (2)** from the vector.
2. **Removing all odd numbers** from the vector.

After these operations, the program prints the updated vector and its size. The code also uses `shrink_to_fit()` to optimize memory usage by reducing the vector's capacity to match its size.

### Problem Being Solved

The problem being solved is **filtering elements from a vector** based on specific criteria. This is a common task in programming where you need to clean or process data by removing unwanted elements. The code shows how to do this efficiently using the C++ Standard Library.

### Approach Taken

The code uses the following key components and algorithms:

1. **`std::vector<int>`**: A dynamic array that stores integers. It allows for efficient insertion, deletion, and traversal of elements.
2. **`std::remove` and `std::remove_if`**: Algorithms from the C++ Standard Library that "remove" elements from a range. These functions do not actually erase elements but instead move the unwanted elements to the end of the range and return an iterator pointing to the new end of the range.
3. **`std::vector::erase`**: Used to actually erase the unwanted elements from the vector.
4. **`std::vector::shrink_to_fit`**: Reduces the vector's capacity to match its size, freeing up unused memory.
5. **Lambda function**: Used to define a custom condition for removing odd numbers.

### How the Code Works Together

1. **Initialization**:
   - A vector `v` is initialized with some integer values: `{1, 2, 3, 2, 5, 2, 6, 2, 4, 8}`.

2. **Removing all occurrences of `2`**:
   - The `std::remove` algorithm is used to move all `2`s to the end of the vector. It returns an iterator (`new_end`) pointing to the new logical end of the vector.
   - The `erase` function is then called to remove the elements from `new_end` to the actual end of the vector.

3. **Printing the intermediate result**:
   - The size of the vector and its contents are printed to show the result after removing `2`s.

4. **Removing all odd numbers**:
   - A lambda function (`odd`) is defined to check if a number is odd.
   - The `std::remove_if` algorithm is used to move all odd numbers to the end of the vector. It returns an iterator (`new_end`) pointing to the new logical end of the vector.
   - The `erase` function is called again to remove the odd numbers.

5. **Optimizing memory usage**:
   - The `shrink_to_fit` function is called to reduce the vector's capacity to match its size, freeing up any unused memory.

6. **Printing the final result**:
   - The size of the vector and its contents are printed again to show the final result after removing odd numbers.

### Overall Structure

The code is structured as follows:

1. **Include necessary headers**:
   - `<iostream>` for input/output operations.
   - `<vector>` for using the `std::vector` container.
   - `<algorithm>` for using the `std::remove` and `std::remove_if` algorithms.

2. **Main function**:
   - Initialize the vector with some values.
   - Remove all occurrences of `2` and print the result.
   - Remove all odd numbers and print the final result.
   - Optimize memory usage with `shrink_to_fit`.

### Summary

This code is a practical example of how to filter elements from a vector in C++. It demonstrates the use of Standard Library algorithms (`remove`, `remove_if`) and vector operations (`erase`, `shrink_to_fit`) to efficiently manipulate data. The code is structured to clearly show the intermediate and final results of the filtering process, making it easy to understand and follow.