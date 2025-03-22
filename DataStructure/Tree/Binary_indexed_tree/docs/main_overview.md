# Code Overview: main.cpp

This C++ code implements a **Fenwick Tree (also known as a Binary Indexed Tree, or BIT)**, which is a data structure used to efficiently perform **range sum queries** and **point updates** on an array. Let’s break down the purpose, functionality, and structure of the code in detail.

---

### **Problem Being Solved**
The code solves the following problem:
- You are given an array of numbers.
- You need to handle two types of operations efficiently:
  1. **Update an element** at a specific index in the array.
  2. **Query the sum of elements** in a range `[a, b]` of the array.

The challenge is to perform these operations quickly, especially when the array is large and there are many operations. A naive approach (e.g., recalculating the sum from scratch for each query) would be too slow for large inputs. The Fenwick Tree is used to optimize these operations.

---

### **Main Functionality**
The code implements the following:
1. **Initialization**:
   - Reads the size of the array (`N`), the number of update operations (`M`), and the number of query operations (`K`).
   - Initializes the array (`inp`) and the Fenwick Tree (`bit`).

2. **Point Updates**:
   - Updates an element in the array and propagates the change through the Fenwick Tree.

3. **Range Sum Queries**:
   - Computes the sum of elements in a range `[a, b]` using the Fenwick Tree.

---

### **Algorithms Used**
1. **Fenwick Tree (Binary Indexed Tree)**:
   - A data structure that allows:
     - **Point updates** in `O(log N)` time.
     - **Range sum queries** in `O(log N)` time.
   - The tree is represented as an array (`bit`), where each index stores a partial sum of the original array.

2. **Key Operations**:
   - **`up(idx, val)`**: Updates the Fenwick Tree by adding `val` to the element at index `idx` and propagates the change to all relevant indices.
   - **`down(idx)`**: Computes the prefix sum (sum of elements from index `1` to `idx`) using the Fenwick Tree.

---

### **Overall Structure**
The code is structured as follows:
1. **Global Variables**:
   - `N`: Size of the array.
   - `inp`: The original array.
   - `bit`: The Fenwick Tree array.

2. **Functions**:
   - `up(idx, val)`: Updates the Fenwick Tree.
   - `down(idx)`: Computes the prefix sum.

3. **Main Function**:
   - Reads input values (`N`, `M`, `K`).
   - Initializes the array and the Fenwick Tree.
   - Processes `M + K` operations (updates and queries).

---

### **How the Code Works Together**
1. **Initialization**:
   - The array `inp` is populated with input values.
   - The Fenwick Tree `bit` is initialized by calling `up(i, inp[i])` for each element in the array.

2. **Operations**:
   - For each operation:
     - If it’s an **update operation** (`cs == 1`):
       - The difference between the new value and the old value is calculated.
       - The array `inp` is updated.
       - The Fenwick Tree is updated using `up(a, diff)`.
     - If it’s a **query operation** (`cs == 2`):
       - The sum of the range `[a, b]` is computed as `down(b) - down(a - 1)`.
       - The result is printed.

---

### **Key Concepts**
1. **Fenwick Tree**:
   - The tree is built by storing partial sums in a way that allows efficient updates and queries.
   - Each index in the tree is responsible for a range of indices in the original array.

2. **Bitwise Operations**:
   - The expressions `idx += (idx & (-idx))` and `idx -= (idx & (-idx))` are used to traverse the tree efficiently.
   - `idx & (-idx)` isolates the least significant bit of `idx`, which determines the range of responsibility for each index.

3. **Efficiency**:
   - Both updates and queries are performed in `O(log N)` time, making the solution suitable for large inputs.

---

### **Example**
Suppose the input is:
```
5 2 2
1 2 3 4 5
1 2 10
2 2 4
1 4 20
2 1 5
```

- The array is initialized as `[1, 2, 3, 4, 5]`.
- The first operation updates the second element to `10`.
- The second operation queries the sum of elements from index `2` to `4` (result: `10 + 3 + 4 = 17`).
- The third operation updates the fourth element to `20`.
- The fourth operation queries the sum of elements from index `1` to `5` (result: `1 + 10 + 3 + 20 + 5 = 39`).

---

### **Summary**
This code efficiently solves the problem of handling point updates and range sum queries using a Fenwick Tree. It demonstrates how to use bitwise operations and partial sums to achieve logarithmic time complexity for both updates and queries. The structure is clean, and the logic is well-organized, making it a good example of how to implement a Fenwick Tree in C++.