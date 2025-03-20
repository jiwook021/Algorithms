# Code Overview: main.c

This C code implements the **Merge Sort** algorithm, which is a popular and efficient sorting algorithm. The purpose of the code is to sort an array of integers in **ascending order**. Let’s break down the main functionality, algorithms used, and the overall structure of the code.

---

### **Problem Being Solved**
The problem being solved is **sorting an array of integers**. Sorting is a fundamental operation in computer science, and Merge Sort is one of the most efficient algorithms for this task. It works by dividing the array into smaller subarrays, sorting them, and then merging them back together in the correct order.

---

### **Approach Taken**
The code uses the **Divide and Conquer** strategy, which is the core idea behind Merge Sort. Here’s how it works:
1. **Divide**: The array is recursively divided into two halves until each subarray contains only one element (or is empty).
2. **Conquer**: The subarrays are sorted individually.
3. **Combine**: The sorted subarrays are merged back together to produce the final sorted array.

This approach ensures that the algorithm is efficient and works well for large datasets.

---

### **Overall Structure**
The code is structured into three main functions:
1. **`merge`**: Combines two sorted subarrays into a single sorted array.
2. **`mergesort`**: Recursively divides the array into smaller subarrays and calls `merge` to combine them.
3. **`Mergesort`**: A wrapper function that initializes a temporary array and calls the main `mergesort` function.
4. **`main`**: The entry point of the program, which initializes an array, sorts it using `Mergesort`, and prints the sorted array.

---

### **How the Code Works Together**
1. The `main` function initializes an array of 8 integers and calls the `Mergesort` function to sort it.
2. The `Mergesort` function creates a temporary array (`tmp`) and calls the `mergesort` function with the appropriate parameters.
3. The `mergesort` function recursively divides the array into smaller subarrays until each subarray contains only one element. It then calls the `merge` function to combine the sorted subarrays.
4. The `merge` function takes two sorted subarrays and merges them into a single sorted array by comparing elements from both subarrays and placing them in the correct order in the original array.

---

### **Algorithms Used**
1. **Merge Sort**: The primary algorithm used for sorting. It has a time complexity of **O(n log n)**, making it efficient for large datasets.
2. **Recursion**: The `mergesort` function calls itself to divide the array into smaller subarrays.
3. **Merging**: The `merge` function combines two sorted subarrays into a single sorted array.

---

### **Key Features of the Code**
1. **Temporary Array**: A temporary array (`tmp`) is used to store elements during the merging process. This avoids modifying the original array directly until the merging is complete.
2. **Recursive Division**: The array is divided into smaller subarrays recursively until the base case (a single element) is reached.
3. **In-Place Sorting**: The sorting is done in-place, meaning the original array is modified directly without requiring additional memory for the final sorted array (except for the temporary array used during merging).

---

### **Example Walkthrough**
Let’s take the example array from the `main` function:
```c
int arr[8] = {10, 2, 3, 4, 5, 6, 7, 3};
```
1. The `Mergesort` function is called, which initializes a temporary array and calls `mergesort` with the full range of the array (`start = 0`, `end = 8`).
2. The `mergesort` function divides the array into two halves:
   - Left half: `{10, 2, 3, 4}`
   - Right half: `{5, 6, 7, 3}`
3. Each half is further divided recursively until each subarray contains only one element.
4. The `merge` function is called to combine the sorted subarrays back into the original array in ascending order.

---

### **Final Output**
After sorting, the array will be:
```c
{2, 3, 3, 4, 5, 6, 7, 10}
```
This is printed by the `main` function using a `for` loop.

---

### **Summary**
The purpose of this code is to sort an array of integers using the Merge Sort algorithm. It works by recursively dividing the array into smaller subarrays, sorting them, and then merging them back together. The code is structured into three main functions (`merge`, `mergesort`, and `Mergesort`) that work together to achieve the sorting. The `main` function initializes the array, calls the sorting function, and prints the sorted result.