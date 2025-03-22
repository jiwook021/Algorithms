# Code Overview: main.c

This C code implements the **Quicksort algorithm**, which is a highly efficient sorting algorithm used to sort elements in an array. Let's break down the purpose, functionality, and structure of the code in detail.

---

### **Purpose of the Code**
The purpose of this code is to sort an array of integers in ascending order using the **Quicksort algorithm**. Quicksort is a **divide-and-conquer algorithm** that works by selecting a "pivot" element from the array and partitioning the other elements into two sub-arrays:
1. Elements less than the pivot.
2. Elements greater than the pivot.

The algorithm then recursively sorts the sub-arrays. This process continues until the entire array is sorted.

---

### **Main Functionality**
1. **Input**: The code starts with an unsorted array of integers: `{3, 2, 4, 1, 7, 6, 5}`.
2. **Sorting**: The array is sorted using the Quicksort algorithm.
3. **Output**: The sorted array is printed to the console.

---

### **Algorithms Used**
The code uses the **Quicksort algorithm**, which consists of the following key steps:
1. **Partitioning**: The array is divided into two parts based on a pivot element. Elements smaller than the pivot are placed on the left, and elements larger than the pivot are placed on the right.
2. **Recursive Sorting**: The algorithm recursively applies the same process to the left and right sub-arrays until the entire array is sorted.

---

### **Overall Structure**
The code is structured into several functions, each with a specific role:
1. **`Swap` Function**: Swaps two elements in the array.
2. **`Partition` Function**: Partitions the array around a pivot element and returns the index of the pivot.
3. **`QuickSort` Function**: Recursively sorts the array by partitioning it and then sorting the sub-arrays.
4. **`main` Function**: Initializes the array, prints the unsorted array, calls the `QuickSort` function, and prints the sorted array.

---

### **How the Code Works Together**
1. **Initialization**:
   - The `main` function initializes an array with unsorted integers: `{3, 2, 4, 1, 7, 6, 5}`.
   - It calculates the length of the array using `sizeof(arr)/sizeof(int)`.

2. **Printing the Unsorted Array**:
   - The `main` function prints the unsorted array to the console.

3. **Sorting the Array**:
   - The `QuickSort` function is called with the array, the starting index (`0`), and the ending index (`len - 1`).
   - Inside `QuickSort`, the `Partition` function is called to divide the array into two parts.
   - The `Partition` function selects the first element as the pivot and rearranges the array so that elements smaller than the pivot are on the left, and elements larger than the pivot are on the right.
   - The `Swap` function is used to swap elements during partitioning.
   - After partitioning, `QuickSort` recursively sorts the left and right sub-arrays.

4. **Printing the Sorted Array**:
   - Once the array is sorted, the `main` function prints the sorted array to the console.

---

### **Key Concepts in the Code**
1. **Divide-and-Conquer**:
   - Quicksort divides the problem into smaller sub-problems (sorting sub-arrays) and solves them recursively.

2. **Pivot Selection**:
   - The pivot is the element around which the array is partitioned. In this implementation, the first element is chosen as the pivot.

3. **Recursion**:
   - The `QuickSort` function calls itself to sort the sub-arrays, which is a hallmark of recursive algorithms.

4. **In-Place Sorting**:
   - Quicksort sorts the array in place, meaning it does not require additional memory for temporary arrays.

---

### **Example Walkthrough**
Let's walk through the sorting process for the array `{3, 2, 4, 1, 7, 6, 5}`:
1. **Initial Array**: `[3, 2, 4, 1, 7, 6, 5]`
2. **First Partition**:
   - Pivot: `3`
   - After partitioning: `[2, 1, 3, 4, 7, 6, 5]`
3. **Recursive Sorting**:
   - Left sub-array: `[2, 1]`
   - Right sub-array: `[4, 7, 6, 5]`
4. **Sorting Left Sub-array**:
   - Pivot: `2`
   - After partitioning: `[1, 2]`
5. **Sorting Right Sub-array**:
   - Pivot: `4`
   - After partitioning: `[4, 5, 6, 7]`
6. **Final Sorted Array**: `[1, 2, 3, 4, 5, 6, 7]`

---

### **Summary**
This code demonstrates how to implement the Quicksort algorithm in C. It uses recursion, partitioning, and swapping to efficiently sort an array of integers. The code is modular, with separate functions for swapping, partitioning, and sorting, making it easy to understand and maintain.

In the next questions, we'll dive deeper into the line-by-line explanation and potential improvements to the code. Let me know when you're ready to proceed!