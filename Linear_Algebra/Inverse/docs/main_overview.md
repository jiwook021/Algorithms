# Code Overview: main.cpp

This C++ code is designed to handle **matrix operations**, specifically focusing on **matrix inversion** using the **Gauss-Jordan elimination** method. Let’s break down the purpose, functionality, and structure of the code in detail:

---

### **Purpose of the Code**
The code defines a `Matrix` class that encapsulates matrix data and provides functionality to:
1. **Store and manage matrix data** (a 2D vector of doubles).
2. **Compute the inverse of a square matrix** using the Gauss-Jordan elimination algorithm.
3. **Print the matrix** in a formatted way for visualization.

The main problem being solved is **matrix inversion**, which is a fundamental operation in linear algebra. Matrix inversion is used in various applications, such as solving systems of linear equations, computer graphics, machine learning, and more.

---

### **Main Functionality**
1. **Matrix Representation**:
   - The matrix is represented as a 2D vector (`std::vector<std::vector<double>>`), where each inner vector represents a row of the matrix.
   - The `Matrix` class provides methods to access the number of rows and columns (`rows()` and `cols()`).

2. **Matrix Inversion**:
   - The `inverse()` method computes the inverse of a square matrix using the **Gauss-Jordan elimination** algorithm.
   - This algorithm transforms the original matrix into its inverse by performing row operations on an **augmented matrix** (a combination of the original matrix and the identity matrix).

3. **Error Handling**:
   - The code includes robust error handling to ensure the matrix is valid (e.g., non-empty, consistent row sizes) and invertible (e.g., square and non-singular).

4. **Matrix Printing**:
   - The `print()` method outputs the matrix to the console in a readable format.

---

### **Algorithms Used**
1. **Gauss-Jordan Elimination**:
   - This is the core algorithm used to compute the inverse of a matrix.
   - It works by transforming the original matrix into the identity matrix while simultaneously applying the same operations to an identity matrix. The result is the inverse of the original matrix.
   - The algorithm involves:
     - **Pivoting**: Selecting the row with the largest absolute value in the current column to avoid division by small numbers (improves numerical stability).
     - **Row Operations**: Swapping rows, scaling rows, and subtracting rows to eliminate entries below and above the pivot.

2. **Augmented Matrix**:
   - The algorithm uses an augmented matrix of size `n x 2n`, where the left half is the original matrix and the right half is the identity matrix.
   - By performing row operations on the augmented matrix, the left half becomes the identity matrix, and the right half becomes the inverse.

---

### **Overall Structure**
The code is organized into the following components:

1. **Matrix Class**:
   - Encapsulates the matrix data and operations.
   - Provides methods for:
     - Initialization (`Matrix()` constructor).
     - Accessing dimensions (`rows()` and `cols()`).
     - Computing the inverse (`inverse()`).
     - Printing the matrix (`print()`).

2. **Constructor**:
   - Validates the input matrix to ensure it is non-empty and has consistent row sizes.
   - Throws an exception if the matrix is invalid.

3. **Inverse Method**:
   - Implements the Gauss-Jordan elimination algorithm.
   - Handles edge cases (e.g., non-square matrices, singular matrices).

4. **Main Function**:
   - Demonstrates the functionality by creating a 2x2 matrix, computing its inverse, and printing both the original and inverse matrices.
   - Includes error handling to catch and report exceptions.

---

### **How the Parts Work Together**
1. **Matrix Initialization**:
   - The `Matrix` constructor validates the input matrix and stores it in the `data_` member variable.

2. **Matrix Inversion**:
   - The `inverse()` method creates an augmented matrix, performs Gauss-Jordan elimination, and extracts the inverse from the augmented matrix.

3. **Matrix Printing**:
   - The `print()` method iterates through the matrix data and outputs it in a formatted way.

4. **Main Function**:
   - Creates a matrix, computes its inverse, and prints both matrices.
   - Handles any errors that occur during matrix operations.

---

### **Example Walkthrough**
For the example matrix:
```
[4, 7]
[2, 6]
```
The code:
1. Creates a `Matrix` object with this data.
2. Computes its inverse using Gauss-Jordan elimination.
3. Prints the original matrix and its inverse.

The inverse of the example matrix is:
```
[ 0.6, -0.7]
[-0.2,  0.4]
```

---

### **Key Takeaways**
- The code solves the problem of **matrix inversion** using a well-known algorithm (Gauss-Jordan elimination).
- It is structured to handle errors gracefully and provides a clean interface for matrix operations.
- The use of an augmented matrix and row operations ensures the inverse is computed efficiently and accurately.

This code is a great example of how to implement a mathematical algorithm in C++ while maintaining readability, modularity, and robustness.