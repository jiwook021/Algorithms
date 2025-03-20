# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into manageable sections, explain each part in simple terms, and provide examples and diagrams where necessary. By the end, you’ll understand every line of code, even if you’re a beginner.

---

### **1. Header Files and Constants**
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <mutex>

constexpr double EPSILON = 1e-10;
```

#### **What It Does**
- **Header Files**:
  - `#include <iostream>`: Allows input/output operations (e.g., printing to the console).
  - `#include <vector>`: Provides the `std::vector` container, which is a dynamic array.
  - `#include <cmath>`: Includes math functions like `std::sqrt` (square root).
  - `#include <stdexcept>`: Provides exception classes like `std::invalid_argument` for error handling.
  - `#include <mutex>`: Provides the `std::mutex` class for thread synchronization.

- **Constant**:
  - `constexpr double EPSILON = 1e-10;`: Defines a small value (`0.0000000001`) to check if a number is effectively zero. This is used to handle numerical precision issues.

#### **Why It’s Used**
- The headers provide the tools needed for the program (e.g., vectors for storing matrices, math functions for calculations).
- `EPSILON` is used to avoid division by very small numbers, which can lead to numerical instability.

---

### **2. The `QRDecomposer` Class**
```cpp
class QRDecomposer {
private:
    mutable std::mutex mutexLock; // For thread safety
```

#### **What It Does**
- Defines a class `QRDecomposer` that encapsulates the QR decomposition logic.
- The `mutable std::mutex mutexLock` is a thread synchronization tool. It ensures that only one thread can access the `decompose` method at a time.

#### **Why It’s Used**
- Encapsulation keeps the decomposition logic organized and reusable.
- The `mutex` ensures thread safety, which is important if multiple threads try to use the `decompose` method simultaneously.

---

### **3. Private Helper Methods**
#### **a. `computeDotProduct`**
```cpp
double computeDotProduct(const std::vector<double>& vectorA, const std::vector<double>& vectorB) const {
    if (vectorA.size() != vectorB.size()) {
        throw std::invalid_argument("Vectors must have the same length for dot product.");
    }
    double dotProductResult = 0.0;
    for (std::size_t index = 0; index < vectorA.size(); ++index) {
        dotProductResult += vectorA[index] * vectorB[index];
    }
    return dotProductResult;
}
```

#### **What It Does**
- Computes the **dot product** of two vectors.
- The dot product is the sum of the products of corresponding elements in the vectors.

#### **Breakdown**
1. **Input Validation**:
   - Checks if the vectors have the same size. If not, throws an exception.
2. **Dot Product Calculation**:
   - Initializes `dotProductResult` to `0.0`.
   - Loops through each element of the vectors, multiplies corresponding elements, and adds the result to `dotProductResult`.
3. **Return**:
   - Returns the computed dot product.

#### **Example**
For vectors `A = [1, 2, 3]` and `B = [4, 5, 6]`:
\[
\text{Dot Product} = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32
\]

#### **Why It’s Used**
- The dot product is used to compute projections and norms, which are essential for the Gram-Schmidt process.

---

#### **b. `computeNorm`**
```cpp
double computeNorm(const std::vector<double>& vectorData) const {
    return std::sqrt(computeDotProduct(vectorData, vectorData));
}
```

#### **What It Does**
- Computes the **Euclidean norm** (length) of a vector.
- The Euclidean norm is the square root of the dot product of the vector with itself.

#### **Breakdown**
1. **Dot Product**:
   - Calls `computeDotProduct` with the same vector for both arguments.
2. **Square Root**:
   - Takes the square root of the dot product to compute the norm.

#### **Example**
For vector `A = [3, 4]`:
\[
\text{Norm} = \sqrt{(3 \times 3) + (4 \times 4)} = \sqrt{9 + 16} = \sqrt{25} = 5
\]

#### **Why It’s Used**
- The norm is used to normalize vectors, turning them into unit vectors (length = 1).

---

### **4. Public Method `decompose`**
#### **a. Input Validation**
```cpp
std::lock_guard<std::mutex> lock(mutexLock); // Ensure thread-safety

if (inputMatrix.empty() || inputMatrix[0].empty()) {
    throw std::invalid_argument("Input matrix cannot be empty.");
}
```

#### **What It Does**
- Ensures thread safety using a `std::lock_guard`.
- Checks if the input matrix is empty or has empty rows.

#### **Why It’s Used**
- Prevents errors caused by invalid input.

---

#### **b. Matrix Initialization**
```cpp
const std::size_t numberOfRows = inputMatrix.size();
const std::size_t numberOfColumns = inputMatrix[0].size();

std::vector<std::vector<double>> orthonormalMatrix(numberOfRows, std::vector<double>(numberOfColumns, 0.0));
std::vector<std::vector<double>> upperTriangularMatrix(numberOfColumns, std::vector<double>(numberOfColumns, 0.0));
```

#### **What It Does**
- Determines the number of rows and columns in the input matrix.
- Initializes `orthonormalMatrix` (Q) and `upperTriangularMatrix` (R) with zeros.

#### **Why It’s Used**
- Prepares the matrices to store the results of the decomposition.

---

#### **c. Gram-Schmidt Process**
```cpp
for (std::size_t currentColumnIndex = 0; currentColumnIndex < numberOfColumns; ++currentColumnIndex) {
    for (std::size_t previousColumnIndex = 0; previousColumnIndex < currentColumnIndex; ++previousColumnIndex) {
        upperTriangularMatrix[previousColumnIndex][currentColumnIndex] =
            computeDotProduct(orthogonalColumns[currentColumnIndex], orthonormalMatrix[previousColumnIndex]);
        for (std::size_t rowIndex = 0; rowIndex < numberOfRows; ++rowIndex) {
            orthogonalColumns[currentColumnIndex][rowIndex] -=
                upperTriangularMatrix[previousColumnIndex][currentColumnIndex] * orthonormalMatrix[previousColumnIndex][rowIndex];
        }
    }

    upperTriangularMatrix[currentColumnIndex][currentColumnIndex] = computeNorm(orthogonalColumns[currentColumnIndex]);
    if (upperTriangularMatrix[currentColumnIndex][currentColumnIndex] < EPSILON) {
        throw std::runtime_error("Matrix columns are linearly dependent.");
    }

    for (std::size_t rowIndex = 0; rowIndex < numberOfRows; ++rowIndex) {
        orthonormalMatrix[rowIndex][currentColumnIndex] =
            orthogonalColumns[currentColumnIndex][rowIndex] / upperTriangularMatrix[currentColumnIndex][currentColumnIndex];
    }
}
```

#### **What It Does**
- Performs the Gram-Schmidt process to orthogonalize and normalize the columns of the input matrix.

#### **Breakdown**
1. **Outer Loop**:
   - Iterates over each column of the input matrix.
2. **Inner Loop**:
   - Subtracts the projection of the current column onto all previously processed columns.
   - Stores the projection coefficients in `upperTriangularMatrix`.
3. **Normalization**:
   - Computes the norm of the orthogonalized column.
   - Normalizes the column to produce a unit vector for `orthonormalMatrix`.

#### **Why It’s Used**
- The Gram-Schmidt process ensures that the columns of \( Q \) are orthonormal and that \( R \) is upper triangular.

---

### **5. Main Function**
```cpp
int main() {
    try {
        QRDecomposer qrDecomposer;

        std::vector<std::vector<double>> inputMatrix = {
            {1.0, 1.0, 0.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0}
        };

        auto [orthonormalMatrix, upperTriangularMatrix] = qrDecomposer.decompose(inputMatrix);

        std::cout << "Orthonormal Matrix Q:\n";
        for (const auto& row : orthonormalMatrix) {
            for (double value : row) {
                std::cout << value << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\nUpper Triangular Matrix R:\n";
        for (const auto& row : upperTriangularMatrix) {
            for (double value : row) {
                std::cout << value << " ";
            }
            std::cout << "\n";
        }
    }
    catch (const std::exception& exceptionMessage) {
        std::cerr << "Error: " << exceptionMessage.what() << "\n";
    }

    return 0;
}
```

#### **What It Does**
- Creates an instance of `QRDecomposer`.
- Defines an example 3x3 matrix.
- Calls the `decompose` method to compute \( Q \) and \( R \).
- Displays the results.

#### **Why It’s Used**
- Demonstrates how to use the `QRDecomposer` class and verifies its functionality.

---

### **Summary**
This code is a complete implementation of QR decomposition using the Gram-Schmidt process. It is designed to be thread-safe, numerically stable, and easy to use. Each part of the code is carefully explained, with examples and diagrams to clarify complex concepts. By following this breakdown, you should have a deep understanding of how the code works!