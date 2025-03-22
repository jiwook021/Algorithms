# Code Overview: main.c

### Purpose of the Code

This C code implements a **Union-Find (Disjoint Set Union, DSU)** data structure, which is a fundamental algorithm used to manage and query disjoint sets efficiently. The primary purpose of this code is to:

1. **Manage disjoint sets**: It allows you to group elements into sets and perform operations like merging sets (union) and checking if two elements belong to the same set (find).
2. **Solve connectivity problems**: It can determine whether two elements are connected (i.e., belong to the same set) and merge sets dynamically.

The code is particularly useful in problems involving **graph connectivity**, such as:
- Detecting cycles in a graph.
- Finding connected components in a graph.
- Solving problems in network connectivity, image processing, and more.

---

### Main Functionality and Algorithms Used

The code uses the **Union-Find algorithm**, which consists of two main operations:
1. **Find**: Determines the root parent of a given element (i.e., the representative of the set to which the element belongs).
2. **Union**: Merges two sets by connecting their root parents.

The algorithm is optimized using **path compression** (in the `getParent` function) to make the `find` operation faster. Path compression ensures that all nodes in a set point directly to the root parent, flattening the structure and reducing the time complexity of future operations.

---

### Overall Structure of the Code

The code is structured into four main parts:
1. **`getParent` Function**: Finds the root parent of an element and applies path compression.
2. **`unionParent` Function**: Merges two sets by connecting their root parents.
3. **`findParent` Function**: Checks if two elements belong to the same set.
4. **`main` Function**: Initializes the data structure, performs union operations, and checks connectivity.

---

### How the Code Works Together

1. **Initialization**:
   - The `main` function initializes an array `parent` of size 11, where each element initially points to itself (`parent[i] = i`). This represents 10 disjoint sets (elements 1 through 10).

2. **Union Operations**:
   - The `unionParent` function is called multiple times to merge sets. For example:
     - `unionParent(parent, 1, 2)` merges the sets containing elements 1 and 2.
     - `unionParent(parent, 2, 3)` merges the sets containing elements 2 and 3, and so on.
   - Each union operation connects the root parents of the two sets, ensuring that all elements in the merged set share the same root.

3. **Find Operations**:
   - The `findParent` function checks if two elements belong to the same set by comparing their root parents.
   - For example, `findParent(parent, 1, 5)` checks if elements 1 and 5 are connected.

4. **Output**:
   - The program prints the results of the `findParent` operations to show whether elements are connected before and after union operations.

---

### Problem Being Solved

The code solves the problem of **dynamic connectivity**, where you need to:
- Efficiently manage a collection of disjoint sets.
- Perform union and find operations dynamically.
- Answer queries about whether two elements are connected.

---

### Approach Taken

The code uses the following approach:
1. **Path Compression**:
   - In the `getParent` function, each call to `getParent` updates the parent of a node to point directly to the root. This flattens the tree structure, making future operations faster.

2. **Union by Size/Rank**:
   - The `unionParent` function connects the smaller tree to the larger tree (based on the root values). This helps keep the tree balanced, improving efficiency.

3. **Efficient Queries**:
   - The `findParent` function uses the optimized `getParent` function to quickly determine if two elements are in the same set.

---

### Example Walkthrough

Let’s walk through the code step by step with the example in the `main` function:

1. **Initialization**:
   - `parent = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` (indices 1-10 represent elements 1-10).

2. **Union Operations**:
   - `unionParent(parent, 1, 2)`:
     - Finds root of 1: 1.
     - Finds root of 2: 2.
     - Connects 2 to 1: `parent[2] = 1`.
     - `parent = [0, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10]`.
   - `unionParent(parent, 2, 3)`:
     - Finds root of 2: 1.
     - Finds root of 3: 3.
     - Connects 3 to 1: `parent[3] = 1`.
     - `parent = [0, 1, 1, 1, 4, 5, 6, 7, 8, 9, 10]`.
   - Similar operations are performed for other unions.

3. **Find Operations**:
   - `findParent(parent, 1, 5)`:
     - Finds root of 1: 1.
     - Finds root of 5: 5.
     - Returns 0 (not connected).
   - After `unionParent(parent, 1, 5)`:
     - Finds root of 1: 1.
     - Finds root of 5: 5.
     - Connects 5 to 1: `parent[5] = 1`.
     - `findParent(parent, 1, 5)` now returns 1 (connected).

---

### Summary

This code demonstrates the Union-Find algorithm, which is a powerful tool for managing disjoint sets and solving connectivity problems. It uses path compression and union by size to ensure efficient operations. The `main` function provides a concrete example of how the algorithm works in practice.

Let me know if you'd like a line-by-line explanation or suggestions for improvements!