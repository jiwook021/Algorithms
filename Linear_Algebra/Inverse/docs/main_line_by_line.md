# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. I’ll also define technical terms and explain the reasoning behind the code’s design.

---

### **1. Includes and Class Definition**
```cpp
#include <iostream>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <utility> // for std::swap
```

#### What It Does:
- These lines include necessary libraries for the program:
  - `<iostream>`: For input/output (e.g., printing to the console).
  - `<vector>`: For using dynamic arrays (vectors) to store matrix data.
  - `<stdexcept>`: For handling exceptions (errors).
  - `<iomanip>`: For formatting output (e.g., setting column widths).
  - `<cmath>`: For mathematical functions (e.g., `abs()` for absolute value).
  - `<utility>`: For `std::swap`, which swaps two values.

#### Why It’s Used:
- These libraries provide the tools needed to work with matrices, handle errors, and format output.

---

### **2. Matrix Class Definition**
```cpp
class Matrix {
public:
    Matrix(const std::vector<std::vector<double>>& elements);
    size_t rows() const;
    size_t cols() const;
    Matrix inverse() const;
    void print() const;

private:
    std::vector<std::vector<double>> data_;
};
```

#### What It Does:
- Defines a `Matrix` class to represent and manipulate matrices.
- **Public Methods**:
  - `Matrix()`: Constructor to initialize the matrix.
  - `rows()`: Returns the number of rows.
  - `cols()`: Returns the number of columns.
  - `inverse()`: Computes the inverse of the matrix.
  - `print()`: Prints the matrix to the console.
- **Private Data**:
  - `data_`: A 2D vector to store the matrix elements.

#### Why It’s Used:
- Encapsulates matrix data and operations into a single class, making the code modular and reusable.

---

### **3. Matrix Constructor**
```cpp
Matrix::Matrix(const std::vector<std::vector<double>>& elements) {
    if (elements.empty() || elements[0].empty()) {
        throw std::invalid_argument("Matrix cannot be empty.");
    }
    size_t columnSize = elements[0].size();
    for (const auto& row : elements) {
        if (row.size() != columnSize) {
            throw std::invalid_argument("All rows must have the same number of columns.");
        }
    }
    data_ = elements;
}
```

#### What It Does:
- Validates the input matrix and stores it in `data_`.
- Checks if the matrix is empty or if rows have inconsistent sizes.
- Throws an exception if the matrix is invalid.

#### Why It’s Used:
- Ensures the matrix is valid before performing operations.

#### Example:
For the input:
```cpp
std::vector<std::vector<double>> matrixElements = {
    {4, 7},
    {2, 6}
};
```
- The constructor checks that:
  1. The matrix is not empty.
  2. All rows have the same number of columns.

---

### **4. Rows and Columns Methods**
```cpp
size_t Matrix::rows() const {
    return data_.size();
}

size_t Matrix::cols() const {
    return data_[0].size();
}
```

#### What It Does:
- `rows()`: Returns the number of rows in the matrix.
- `cols()`: Returns the number of columns in the matrix.

#### Why It’s Used:
- Provides easy access to the matrix dimensions.

---

### **5. Print Method**
```cpp
void Matrix::print() const {
    for (const auto& row : data_) {
        for (double value : row) {
            std::cout << std::setw(10) << value << " ";
        }
        std::cout << "\n";
    }
}
```

#### What It Does:
- Iterates through the matrix and prints each element with formatting.

#### Why It’s Used:
- Allows visualization of the matrix.

#### Example:
For the matrix:
```
[4, 7]
[2, 6]
```
The output will be:
```
         4          7 
         2          6 
```

---

### **6. Inverse Method**
```cpp
Matrix Matrix::inverse() const {
    if (rows() != cols()) {
        throw std::runtime_error("Only square matrices can be inverted.");
    }
    size_t n = rows();

    // Create augmented matrix [A | I]
    std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented[i][j] = data_[i][j];
        }
        augmented[i][n + i] = 1.0; // Set identity matrix.
    }

    // Perform Gauss-Jordan elimination
    for (size_t i = 0; i < n; i++) {
        // Pivot selection
        size_t pivotRow = i;
        double maxElement = std::abs(augmented[i][i]);
        for (size_t k = i + 1; k < n; k++) {
            if (std::abs(augmented[k][i]) > maxElement) {
                maxElement = std::abs(augmented[k][i]);
                pivotRow = k;
            }
        }

        // Check for singularity
        if (std::abs(augmented[pivotRow][i]) < 1e-12) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }

        // Swap rows if necessary
        if (pivotRow != i) {
            std::swap(augmented[i], augmented[pivotRow]);
        }

        // Normalize pivot row
        double pivotValue = augmented[i][i];
        for (size_t j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivotValue;
        }

        // Eliminate other entries in column i
        for (size_t row = 0; row < n; row++) {
            if (row != i) {
                double factor = augmented[row][i];
                for (size_t col = 0; col < 2 * n; col++) {
                    augmented[row][col] -= factor * augmented[i][col];
                }
            }
        }
    }

    // Extract inverse matrix
    std::vector<std::vector<double>> inverseData(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            inverseData[i][j] = augmented[i][n + j];
        }
    }

    return Matrix(inverseData);
}
```

#### What It Does:
- Computes the inverse of a square matrix using Gauss-Jordan elimination.

#### Why It’s Used:
- Matrix inversion is a fundamental operation in linear algebra.

#### Example:
For the matrix:
```
[4, 7]
[2, 6]
```
The inverse is:
```
[ 0.6, -0.7]
[-0.2,  0.4]
```

---

### **7. Main Function**
```cpp
int main() {
    try {
        std::vector<std::vector<double>> matrixElements = {
            {4, 7},
            {2, 6}
        };

        Matrix matrix(matrixElements);

        std::cout << "Original Matrix:\n";
        matrix.print();

        Matrix inverseMatrix = matrix.inverse();

        std::cout << "\nInverse Matrix:\n";
        inverseMatrix.print();
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

#### What It Does:
- Demonstrates the functionality by creating a matrix, computing its inverse, and printing both.

#### Why It’s Used:
- Provides a simple way to test the `Matrix` class.

---

This explanation should make the code accessible to everyone, from beginners to experts! Let me know if you have further questions.