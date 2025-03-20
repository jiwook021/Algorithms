# Code Overview: main.c

### Purpose and Main Functionality of the Code

This C code is designed to sort an array of integers in **ascending order** using the **Selection Sort** algorithm. The code consists of two main parts:

1. **`SelSort` Function**: This is where the sorting logic is implemented. It takes an array of integers and its size as input and sorts the array in place (meaning it modifies the original array directly).

2. **`main` Function**: This is the entry point of the program. It initializes an array of integers, calls the `SelSort` function to sort the array, and then prints the sorted array to the console.

---

### Problem Being Solved

The problem being solved is **sorting an array of integers**. Sorting is a fundamental operation in computer science, and it involves arranging elements in a specific order (in this case, ascending order). The Selection Sort algorithm is used here to achieve this.

---

### Approach Taken: Selection Sort Algorithm

The **Selection Sort** algorithm works by repeatedly finding the smallest (or largest, depending on the order) element in the unsorted portion of the array and swapping it with the first element of the unsorted portion. This process continues until the entire array is sorted.

Here’s how the algorithm works step-by-step:

1. **Divide the Array**: The array is conceptually divided into two parts:
   - **Sorted Subarray**: Initially empty, but grows as the algorithm progresses.
   - **Unsorted Subarray**: Initially the entire array, but shrinks as the algorithm progresses.

2. **Find the Minimum Element**: In each iteration, the algorithm scans the unsorted subarray to find the smallest element.

3. **Swap**: The smallest element found is swapped with the first element of the unsorted subarray. This effectively moves the smallest element to its correct position in the sorted subarray.

4. **Repeat**: The process is repeated for the remaining unsorted subarray until the entire array is sorted.

---

### How the Code Works Together

1. **`SelSort` Function**:
   - The function takes two arguments: the array (`arr`) and its size (`n`).
   - It uses nested loops:
     - The outer loop (`for(i=0; i<n-1; i++)`) iterates over the array, treating each position as the starting point of the unsorted subarray.
     - The inner loop (`for(j=i+1; j<n; j++)`) finds the smallest element in the unsorted subarray.
     - Once the smallest element is found, it is swapped with the first element of the unsorted subarray.

2. **`main` Function**:
   - An array `arr` is initialized with the values `{3, 4, 2, 1}`.
   - The `SelSort` function is called with the array and its size (`sizeof(arr)/sizeof(int)` calculates the number of elements in the array).
   - After sorting, the sorted array is printed using a `for` loop.

---

### Example Walkthrough

Let’s walk through the sorting process with the example array `{3, 4, 2, 1}`:

1. **First Iteration (i = 0)**:
   - Unsorted subarray: `{3, 4, 2, 1}`
   - Find the smallest element: `1` (at index 3).
   - Swap `3` (index 0) with `1` (index 3).
   - Array after swap: `{1, 4, 2, 3}`.

2. **Second Iteration (i = 1)**:
   - Unsorted subarray: `{4, 2, 3}`
   - Find the smallest element: `2` (at index 2).
   - Swap `4` (index 1) with `2` (index 2).
   - Array after swap: `{1, 2, 4, 3}`.

3. **Third Iteration (i = 2)**:
   - Unsorted subarray: `{4, 3}`
   - Find the smallest element: `3` (at index 3).
   - Swap `4` (index 2) with `3` (index 3).
   - Array after swap: `{1, 2, 3, 4}`.

4. **Fourth Iteration (i = 3)**:
   - Unsorted subarray: `{4}`
   - Only one element remains, so no further action is needed.

The final sorted array is `{1, 2, 3, 4}`.

---

### Overall Structure

1. **Input**: An unsorted array of integers.
2. **Processing**: The `SelSort` function sorts the array using the Selection Sort algorithm.
3. **Output**: The sorted array is printed to the console.

---

### Key Takeaways

- The **Selection Sort** algorithm is simple but not the most efficient for large datasets (its time complexity is **O(n²)**).
- The code demonstrates how to:
  - Implement a sorting algorithm.
  - Use nested loops to traverse and manipulate arrays.
  - Swap elements in an array.
  - Print array elements to the console.

This code is a great example of how to solve a basic sorting problem using a straightforward algorithm. In the next questions, we’ll dive deeper into the line-by-line explanation and potential improvements!