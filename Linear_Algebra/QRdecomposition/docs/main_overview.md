# Code Overview: main.cpp

This C++ code implements a **QR decomposition** algorithm using the **Gram-Schmidt process**. Let's break down its purpose, functionality, and structure in detail.

---

### **Purpose of the Code**
The purpose of this code is to perform **QR decomposition** on a given matrix. QR decomposition is a fundamental linear algebra operation that factorizes a matrix \( A \) into two matrices:
1. **\( Q \)**: An orthonormal matrix (columns are orthogonal unit vectors).
2. **\( R \)**: An upper triangular matrix.

The decomposition satisfies the equation:
\[
A = Q \cdot R
\]

QR decomposition is widely used in numerical linear algebra for solving systems of linear equations, computing eigenvalues, and performing least-squares regression.

---

### **Main Functionality**
The code implements the **Gram-Schmidt process**, which is a method for orthogonalizing a set of vectors (the columns of the input matrix) and then normalizing them to produce the orthonormal matrix \( Q \). The upper triangular matrix \( R \) is computed as a byproduct of this process.

---

### **Algorithms Used**
1. **Gram-Schmidt Process**:
   - The Gram-Schmidt process takes the columns of the input matrix and orthogonalizes them one by one.
   - For each column, it subtracts the projections of the column onto all previously processed columns to ensure orthogonality.
   - The resulting orthogonal vectors are then normalized to produce the orthonormal columns of \( Q \).
   - The coefficients used in the projection steps are stored in the upper triangular matrix \( R \).

2. **Dot Product and Norm Calculation**:
   - The dot product is used to compute projections and norms.
   - The Euclidean norm (length) of a vector is calculated using the dot product of the vector with itself.

3. **Thread Safety**:
   - A `std::mutex` is used to ensure that the decomposition process is thread-safe, meaning it can be safely used in a multi-threaded environment.

---

### **Overall Structure**
The code is structured into a class `QRDecomposer` that encapsulates the decomposition logic. Here's how the different parts of the code work together:

1. **Class `QRDecomposer`**:
   - Contains private helper methods (`computeDotProduct` and `computeNorm`) for vector operations.
   - The main method `decompose` performs the QR decomposition using the Gram-Schmidt process.

2. **Private Helper Methods**:
   - `computeDotProduct`: Computes the dot product of two vectors.
   - `computeNorm`: Computes the Euclidean norm of a vector.

3. **Public Method `decompose`**:
   - Takes an input matrix and returns a pair of matrices \( Q \) and \( R \).
   - Ensures thread safety using a `std::mutex`.
   - Validates the input matrix to ensure it is not empty.
   - Initializes the matrices \( Q \) and \( R \).
   - Performs the Gram-Schmidt process to compute \( Q \) and \( R \).

4. **Main Function**:
   - Creates an instance of `QRDecomposer`.
   - Defines an example 3x3 matrix.
   - Calls the `decompose` method to compute \( Q \) and \( R \).
   - Displays the results.

---

### **How the Code Works Together**
1. **Input Matrix**:
   - The input matrix is provided as a 2D vector of doubles.

2. **Initialization**:
   - The matrices \( Q \) and \( R \) are initialized with zeros.
   - A temporary storage `orthogonalColumns` is created to hold intermediate orthogonalized vectors.

3. **Gram-Schmidt Process**:
   - For each column of the input matrix:
     - The column is orthogonalized by subtracting its projections onto all previously processed columns.
     - The coefficients of these projections are stored in \( R \).
     - The orthogonalized column is normalized to produce a column of \( Q \).

4. **Output**:
   - The orthonormal matrix \( Q \) and the upper triangular matrix \( R \) are returned and displayed.

---

### **Problem Being Solved**
The problem being solved is the decomposition of a matrix into its \( Q \) and \( R \) components. This is useful in many applications, such as:
- Solving linear systems \( Ax = b \) more efficiently.
- Computing eigenvalues and eigenvectors.
- Performing least-squares regression.

---

### **Approach Taken**
The code takes a **numerical approach** to solve the problem:
1. It uses the Gram-Schmidt process, which is a straightforward but numerically unstable method for QR decomposition.
2. It ensures thread safety using a mutex, making it suitable for multi-threaded environments.
3. It includes error handling for invalid inputs (e.g., empty matrices or linearly dependent columns).

---

### **Key Components**
1. **Thread Safety**:
   - The `std::mutex` ensures that only one thread can execute the `decompose` method at a time.

2. **Error Handling**:
   - The code checks for invalid inputs (e.g., empty matrices or vectors of unequal length) and throws exceptions with descriptive messages.

3. **Numerical Stability**:
   - The code uses a small constant `EPSILON` to check for linear dependence among columns, ensuring numerical stability.

---

### **Example Input and Output**
#### Input Matrix:
\[
A = \begin{bmatrix}
1.0 & 1.0 & 0.0 \\
1.0 & 0.0 & 1.0 \\
0.0 & 1.0 & 1.0
\end{bmatrix}
\]

#### Output:
- **Orthonormal Matrix \( Q \)**:
  \[
  Q = \begin{bmatrix}
  \text{Orthonormal columns}
  \end{bmatrix}
  \]
- **Upper Triangular Matrix \( R \)**:
  \[
  R = \begin{bmatrix}
  \text{Upper triangular values}
  \end{bmatrix}
  \]

---

### **Summary**
This code is a well-structured implementation of QR decomposition using the Gram-Schmidt process. It is designed to be thread-safe, numerically stable, and easy to use. The decomposition is performed step-by-step, with clear separation of concerns between vector operations, matrix operations, and error handling. The main function demonstrates how to use the `QRDecomposer` class with an example matrix and displays the results.