# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step** in an extremely detailed and beginner-friendly way. I’ll explain every significant section, including the logic, control flow, and technical terms. I’ll also use examples and diagrams to make everything clear.

---

### **1. Header Files and Type Aliases**
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
```

#### **What It Does**
- **Header Files**:
  - `#include <iostream>`: Provides input/output functionality (e.g., printing to the console).
  - `#include <vector>`: Provides the `std::vector` container, which is a dynamic array.
  - `#include <cmath>`: Provides mathematical functions like `std::sqrt` (square root).
  - `#include <stdexcept>`: Provides exception classes like `std::invalid_argument` for error handling.
  - `#include <limits>`: Provides constants for numeric limits (e.g., the smallest possible double value).

- **Type Aliases**:
  - `using Matrix = std::vector<std::vector<double>>;`: Defines `Matrix` as a 2D vector of doubles (a table of numbers).
  - `using Vector = std::vector<double>;`: Defines `Vector` as a 1D vector of doubles (a list of numbers).

#### **Why It’s Used**
- **Header Files**: These are necessary to use specific functionality in the program (e.g., printing, math operations, error handling).
- **Type Aliases**: They make the code more readable by giving meaningful names to complex types.

---

### **2. Helper Function: `dotProduct`**
```cpp
double dotProduct(const Vector& a, const Vector& b) {
    if(a.size() != b.size())
        throw std::invalid_argument("Vector sizes do not match for dot product.");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += a[i] * b[i];
    return sum;
}
```

#### **What It Does**
- Computes the **dot product** of two vectors `a` and `b`.
- The dot product is the sum of the products of corresponding elements in the vectors.

#### **Step-by-Step Explanation**
1. **Input Validation**:
   - Checks if the sizes of `a` and `b` are equal using `a.size() != b.size()`.
   - If they are not equal, throws an exception (`std::invalid_argument`) with an error message.

2. **Initialization**:
   - Declares a variable `sum` and initializes it to `0.0`. This will store the result.

3. **Loop**:
   - A `for` loop iterates over each element of the vectors:
     - `size_t i = 0`: Starts at the first index.
     - `i < a.size()`: Continues until the end of the vector.
     - `++i`: Moves to the next index.
   - Inside the loop:
     - Multiplies the corresponding elements (`a[i] * b[i]`) and adds the result to `sum`.

4. **Return**:
   - Returns the computed `sum`.

#### **Example**
If `a = {1, 2, 3}` and `b = {4, 5, 6}`:
- `sum = (1 * 4) + (2 * 5) + (3 * 6) = 4 + 10 + 18 = 32`.

#### **Why It’s Used**
- The dot product is a fundamental operation in linear algebra, used in many algorithms (e.g., computing norms, projections).

---

### **3. Helper Function: `vectorNorm`**
```cpp
double vectorNorm(const Vector& a) {
    return std::sqrt(dotProduct(a, a));
}
```

#### **What It Does**
- Computes the **Euclidean norm** (also called the **L2 norm**) of a vector `a`.
- The Euclidean norm is the square root of the sum of the squares of the vector’s elements.

#### **Step-by-Step Explanation**
1. **Call `dotProduct`**:
   - Computes the dot product of `a` with itself (`dotProduct(a, a)`), which is the sum of the squares of its elements.

2. **Square Root**:
   - Takes the square root of the result using `std::sqrt`.

3. **Return**:
   - Returns the computed norm.

#### **Example**
If `a = {3, 4}`:
- `dotProduct(a, a) = (3 * 3) + (4 * 4) = 9 + 16 = 25`.
- `vectorNorm(a) = sqrt(25) = 5`.

#### **Why It’s Used**
- The Euclidean norm measures the "length" of a vector, which is useful in many applications (e.g., normalization, distance calculations).

---

### **4. Helper Function: `getColumn`**
```cpp
Vector getColumn(const Matrix &A, size_t j) {
    Vector col;
    for (const auto &row : A) {
        if(j >= row.size())
            throw std::invalid_argument("Column index out of bounds in getColumn.");
        col.push_back(row[j]);
    }
    return col;
}
```

#### **What It Does**
- Extracts the `j`-th column from a matrix `A`.

#### **Step-by-Step Explanation**
1. **Initialization**:
   - Declares an empty vector `col` to store the column.

2. **Loop**:
   - A `for` loop iterates over each row of the matrix:
     - `const auto &row : A`: Accesses each row as a reference.
   - Inside the loop:
     - Checks if the column index `j` is out of bounds (`j >= row.size()`).
     - If out of bounds, throws an exception.
     - Otherwise, appends the `j`-th element of the row to `col` using `push_back`.

3. **Return**:
   - Returns the extracted column.

#### **Example**
If `A = {{1, 2}, {3, 4}}` and `j = 1`:
- `col = {2, 4}`.

#### **Why It’s Used**
- Extracting columns is a common operation in linear algebra (e.g., matrix multiplication, QR decomposition).

---

### **5. Helper Function: `createZeroMatrix`**
```cpp
Matrix createZeroMatrix(size_t n) {
    return Matrix(n, Vector(n, 0.0));
}
```

#### **What It Does**
- Creates an `n x n` matrix filled with zeros.

#### **Step-by-Step Explanation**
1. **Initialization**:
   - Uses the `Matrix` constructor to create a matrix with `n` rows.
   - Each row is a `Vector` of size `n`, initialized to `0.0`.

2. **Return**:
   - Returns the zero matrix.

#### **Example**
If `n = 2`:
- The matrix is `{{0, 0}, {0, 0}}`.

#### **Why It’s Used**
- Zero matrices are often used as placeholders or initial values in algorithms.

---

### **6. Main Function**
```cpp
int main() {
    try {
        // Define a sample non-symmetric 3x3 matrix.
        Matrix sampleMatrix = {
            { 4.0, -2.0,  1.0 },
            { 3.0,  6.0,  2.0 },
            { 2.0,  1.0,  3.0 }
        };

        std::cout << "Original Matrix:\n";
        printMatrix(sampleMatrix);
        std::cout << "\n";

        // Compute the eigendecomposition using the QR algorithm.
        auto [eigenvalues, eigenvectors] = qrEigenDecomposition(sampleMatrix, 1e-10, 1000);

        std::cout << "Eigenvalues (approximate):\n";
        printVector(eigenvalues);
        std::cout << "\nEigenvectors (approximate, each column is an eigenvector):\n";
        printMatrix(eigenvectors);
    }
    catch (const std::exception &ex) {
        std::cerr << "Error during eigendecomposition: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

#### **What It Does**
- Defines a sample matrix, computes its eigenvalues and eigenvectors using the QR algorithm, and prints the results.

#### **Step-by-Step Explanation**
1. **Sample Matrix**:
   - Defines a 3x3 matrix `sampleMatrix`.

2. **Print Original Matrix**:
   - Calls `printMatrix` (not shown) to display the matrix.

3. **Eigendecomposition**:
   - Calls `qrEigenDecomposition` (not shown) to compute the eigenvalues and eigenvectors.
   - Uses structured bindings (`auto [eigenvalues, eigenvectors]`) to store the results.

4. **Print Results**:
   - Calls `printVector` and `printMatrix` to display the eigenvalues and eigenvectors.

5. **Error Handling**:
   - If an exception occurs, it catches the error and prints a message.

#### **Why It’s Used**
- The main function ties everything together, demonstrating the use of the helper functions and the QR algorithm.

---

### **Summary**
This code is a modular implementation of the QR algorithm for eigendecomposition. It uses helper functions for common linear algebra operations and handles errors gracefully. The main function demonstrates the algorithm on a sample matrix and prints the results.

Let me know if you’d like to dive deeper into any specific part!