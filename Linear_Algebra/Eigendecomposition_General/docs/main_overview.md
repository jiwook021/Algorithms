# Code Overview: main.cpp

This C++ code is designed to perform **eigendecomposition** of a square matrix using the **QR algorithm**. Eigendecomposition is a fundamental operation in linear algebra that decomposes a matrix into its eigenvalues and eigenvectors. These are crucial in many areas of mathematics, physics, engineering, and data science, as they provide insights into the properties of linear transformations represented by the matrix.

---

### **Purpose of the Code**
The code aims to:
1. **Compute the eigenvalues and eigenvectors** of a given square matrix.
2. Use the **QR algorithm**, an iterative numerical method, to approximate the eigenvalues and eigenvectors.
3. Provide a modular and reusable implementation with helper functions for common linear algebra operations (e.g., dot product, vector norm, matrix column extraction).

---

### **Main Functionality**
1. **Eigendecomposition**:
   - The code decomposes a matrix into its eigenvalues and eigenvectors.
   - Eigenvalues are scalars that represent how much the eigenvectors are scaled during the linear transformation.
   - Eigenvectors are non-zero vectors that only change by a scalar factor when the matrix is applied to them.

2. **QR Algorithm**:
   - The QR algorithm is an iterative method that repeatedly applies the QR decomposition to a matrix.
   - QR decomposition factors a matrix into an orthogonal matrix \( Q \) and an upper triangular matrix \( R \).
   - By iteratively applying QR decomposition and reconstructing the matrix, the algorithm converges to a diagonal matrix containing the eigenvalues.

---

### **Overall Structure**
The code is structured into several parts:
1. **Helper Functions**:
   - `dotProduct`: Computes the dot product of two vectors.
   - `vectorNorm`: Computes the Euclidean norm (L2 norm) of a vector.
   - `getColumn`: Extracts a specific column from a matrix.
   - `createZeroMatrix`: Creates a square matrix filled with zeros.

2. **Main Function**:
   - Defines a sample 3x3 matrix.
   - Calls the `qrEigenDecomposition` function (not shown in the provided code) to compute the eigenvalues and eigenvectors.
   - Prints the original matrix, eigenvalues, and eigenvectors.

3. **Error Handling**:
   - The code uses `try-catch` blocks to handle exceptions, ensuring robustness.

---

### **How the Code Works Together**
1. **Input**:
   - A sample 3x3 matrix is defined in the `main` function.

2. **Computation**:
   - The `qrEigenDecomposition` function (not shown) is called to compute the eigenvalues and eigenvectors using the QR algorithm.
   - This function likely uses the helper functions (`dotProduct`, `vectorNorm`, `getColumn`, etc.) to perform intermediate calculations.

3. **Output**:
   - The original matrix, eigenvalues, and eigenvectors are printed to the console.

4. **Error Handling**:
   - If any errors occur (e.g., invalid matrix dimensions), the program catches the exception and prints an error message.

---

### **Algorithms Used**
1. **QR Algorithm**:
   - The QR algorithm is the core method used for eigendecomposition.
   - It works by repeatedly decomposing the matrix into \( Q \) (orthogonal) and \( R \) (upper triangular) and reconstructing the matrix as \( RQ \).
   - Over iterations, the matrix converges to a diagonal matrix containing the eigenvalues.

2. **Helper Algorithms**:
   - **Dot Product**: Computes the sum of the products of corresponding elements in two vectors.
   - **Vector Norm**: Computes the magnitude of a vector using the Euclidean norm.
   - **Column Extraction**: Retrieves a specific column from a matrix.

---

### **Problem Being Solved**
The code solves the problem of finding the eigenvalues and eigenvectors of a square matrix. This is important because:
- Eigenvalues and eigenvectors reveal the intrinsic properties of a matrix.
- They are used in applications like principal component analysis (PCA), solving differential equations, and stability analysis in dynamical systems.

---

### **Approach Taken**
1. **Modular Design**:
   - The code is modular, with helper functions for common linear algebra operations.
   - This makes the code reusable and easier to maintain.

2. **Iterative Numerical Method**:
   - The QR algorithm is an iterative method, meaning it approximates the solution over multiple steps.
   - The algorithm is robust and widely used for eigendecomposition.

3. **Error Handling**:
   - The code includes checks for invalid inputs (e.g., mismatched vector sizes, out-of-bounds column indices) and handles exceptions gracefully.

---

### **Summary**
This code is a well-structured implementation of the QR algorithm for eigendecomposition. It uses helper functions to perform common linear algebra operations and handles errors robustly. The main goal is to compute the eigenvalues and eigenvectors of a given square matrix, which are essential for understanding the matrix's properties and behavior in various applications.

Let me know if you'd like a detailed line-by-line explanation or suggestions for improvements!