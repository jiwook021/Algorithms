# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple language, define technical terms, and provide examples to make everything clear. We’ll also explore the **why** behind the design choices.

---

### **1. Header Files**
```cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <cmath>
#include <numeric>
```

#### **What It Does**
These are **header files** that provide functionality for:
- **Input/output** (`<iostream>`): For printing to the console.
- **Vectors** (`<vector>`): For storing the matrix as a 2D array.
- **Random number generation** (`<random>`): For filling the matrix with random values.
- **Time utilities** (`<chrono>`): For timing operations (not used in this snippet but often useful).
- **Formatting output** (`<iomanip>`): For controlling how numbers are displayed (e.g., decimal places).
- **Threading** (`<thread>` and `<mutex>`): For handling multi-threaded operations safely.
- **Mathematical functions** (`<cmath>`): For operations like square roots.
- **Numeric algorithms** (`<numeric>`): For operations like summing elements.

#### **Why These Are Used**
- The code needs to handle **matrices**, which are naturally represented as 2D arrays. Vectors are a flexible and safe way to do this in C++.
- **Random numbers** are used to generate the matrix values.
- **Threading and mutexes** ensure that the code works correctly in multi-threaded environments.
- **Mathematical functions** are needed for operations like calculating norms or determinants.

---

### **2. Class Definition**
```cpp
class SymmetryMatrix {
private:
    std::vector<std::vector<double>> matrix;
    size_t size;
    mutable std::mutex mtx;
```

#### **What It Does**
This defines a class called `SymmetryMatrix`. A **class** is a blueprint for creating objects. Here, the class represents a symmetric matrix.

#### **Key Components**
1. **`matrix`**: A 2D vector (a vector of vectors) to store the matrix values. Each inner vector represents a row of the matrix.
   - Example: For a 3x3 matrix, `matrix` might look like this:
     ```
     matrix = {
         {1.0, 2.0, 3.0},
         {2.0, 4.0, 5.0},
         {3.0, 5.0, 6.0}
     }
     ```
2. **`size`**: The size of the matrix (number of rows or columns, since it’s square).
3. **`mtx`**: A **mutex** (short for "mutual exclusion") used to ensure thread safety. It prevents multiple threads from accessing the matrix at the same time, which could cause errors.

#### **Why These Are Used**
- **`matrix`**: Vectors are dynamic and easy to work with in C++. They automatically handle memory allocation and resizing.
- **`size`**: Storing the size separately makes it easier to work with the matrix.
- **`mtx`**: In multi-threaded programs, mutexes are essential to prevent **race conditions**, where two threads try to modify the same data simultaneously.

---

### **3. Constructor**
```cpp
SymmetryMatrix(size_t n) : size(n) {
    if (n == 0) {
        throw std::invalid_argument("행렬 크기는 0보다 커야 합니다.");
    }
    
    matrix.resize(n, std::vector<double>(n, 0.0));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            double value = dist(gen);
            matrix[i][j] = value;
            if (i != j) {
                matrix[j][i] = value;
            }
        }
    }
}
```

#### **What It Does**
This is the **constructor** for the `SymmetryMatrix` class. It creates a random symmetric matrix of size `n x n`.

#### **Step-by-Step Breakdown**
1. **Check for Valid Size**:
   - If `n` is 0, the code throws an exception. A matrix cannot have a size of 0.
   - Example: If `n = 0`, the program stops and displays an error message.

2. **Resize the Matrix**:
   - The `matrix` is resized to `n x n`, with all elements initialized to `0.0`.
   - Example: If `n = 3`, `matrix` becomes:
     ```
     matrix = {
         {0.0, 0.0, 0.0},
         {0.0, 0.0, 0.0},
         {0.0, 0.0, 0.0}
     }
     ```

3. **Random Number Generation**:
   - A random number generator is created using `std::random_device` and `std::mt19937`.
   - A uniform distribution (`std::uniform_real_distribution`) is used to generate random numbers between `-10.0` and `10.0`.

4. **Fill the Matrix**:
   - The outer loop (`i`) iterates over the rows.
   - The inner loop (`j`) iterates over the columns, starting from `i` (to ensure symmetry).
   - A random value is generated and assigned to `matrix[i][j]`.
   - If `i != j`, the same value is copied to `matrix[j][i]` to enforce symmetry.
   - Example: For `n = 2`, the matrix might look like:
     ```
     matrix = {
         {3.5, 7.2},
         {7.2, -4.1}
     }
     ```

#### **Why This Approach Is Used**
- **Symmetry Enforcement**: By only filling the upper triangle and copying values to the lower triangle, the code ensures the matrix is symmetric.
- **Random Values**: Random numbers make the matrix more interesting and realistic for testing.

---

### **4. Copy Constructor**
```cpp
SymmetryMatrix(const SymmetryMatrix& other) {
    std::lock_guard<std::mutex> lock(other.mtx);
    matrix = other.matrix;
    size = other.size;
}
```

#### **What It Does**
This is the **copy constructor**. It creates a new `SymmetryMatrix` object as a copy of an existing one.

#### **Step-by-Step Breakdown**
1. **Lock the Mutex**:
   - A `std::lock_guard` locks the mutex of the `other` object to ensure thread safety.
   - The lock is automatically released when the `lock_guard` goes out of scope.

2. **Copy the Matrix and Size**:
   - The `matrix` and `size` of the `other` object are copied to the new object.

#### **Why This Is Used**
- **Thread Safety**: The mutex ensures that the `other` object is not modified while it’s being copied.
- **Deep Copy**: The new object gets its own copy of the matrix, so changes to one object don’t affect the other.

---

### **5. Assignment Operator**
```cpp
SymmetryMatrix& operator=(const SymmetryMatrix& other) {
    if (this != &other) {
        std::lock_guard<std::mutex> lock1(mtx);
        std::lock_guard<std::mutex> lock2(other.mtx);
        matrix = other.matrix;
        size = other.size;
    }
    return *this;
}
```

#### **What It Does**
This is the **assignment operator**. It allows one `SymmetryMatrix` object to be assigned to another.

#### **Step-by-Step Breakdown**
1. **Check for Self-Assignment**:
   - If the object is being assigned to itself (`this == &other`), the function returns early to avoid unnecessary work.

2. **Lock Both Mutexes**:
   - Both the current object’s mutex (`mtx`) and the `other` object’s mutex are locked to ensure thread safety.

3. **Copy the Matrix and Size**:
   - The `matrix` and `size` of the `other` object are copied to the current object.

4. **Return the Current Object**:
   - The function returns `*this` to allow chaining (e.g., `a = b = c`).

#### **Why This Is Used**
- **Thread Safety**: Both mutexes are locked to prevent race conditions.
- **Self-Assignment Check**: Prevents unnecessary copying if an object is assigned to itself.

---

### **6. `setValue` Method**
```cpp
void setValue(size_t i, size_t j, double value) {
    std::lock_guard<std::mutex> lock(mtx);
    if (i >= size || j >= size) {
        throw std::out_of_range("인덱스가 범위를 벗어났습니다.");
    }
    matrix[i][j] = value;
}
```

#### **What It Does**
This method sets the value of a specific element in the matrix.

#### **Step-by-Step Breakdown**
1. **Lock the Mutex**:
   - The mutex is locked to ensure thread safety.

2. **Check Index Bounds**:
   - If `i` or `j` is out of bounds (greater than or equal to `size`), an exception is thrown.

3. **Set the Value**:
   - The value at `matrix[i][j]` is updated.

#### **Why This Is Used**
- **Thread Safety**: The mutex ensures that only one thread can modify the matrix at a time.
- **Bounds Checking**: Prevents invalid memory access.

---

### **7. `print` Method**
```cpp
void print() const {
    std::lock_guard<std::mutex> lock(mtx);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
```

#### **What It Does**
This method prints the matrix to the console.

#### **Step-by-Step Breakdown**
1. **Lock the Mutex**:
   - The mutex is locked to ensure thread safety.

2. **Nested Loops**:
   - The outer loop (`i`) iterates over the rows.
   - The inner loop (`j`) iterates over the columns.

3. **Formatting**:
   - `std::setw(10)`: Sets the width of each output to 10 characters.
   - `std::fixed` and `std::setprecision(2)`: Ensures numbers are displayed with 2 decimal places.

4. **Print the Value**:
   - The value at `matrix[i][j]` is printed.

5. **Newline After Each Row**:
   - After printing a row, a newline is added.

#### **Why This Is Used**
- **Thread Safety**: The mutex ensures that the matrix is not modified while it’s being printed.
- **Formatting**: Makes the output easier to read.

---

### **8. Main Function**
```cpp
int main() {
    try {
        constexpr size_t MATRIX_SIZE = 3;
        SymmetryMatrix A(MATRIX_SIZE);
        A.print();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
```

#### **What It Does**
This is the **entry point** of the program. It creates a `SymmetryMatrix` object and prints it.

#### **Step-by-Step Breakdown**
1. **Try Block**:
   - The code is wrapped in a `try` block to catch any exceptions.

2. **Create a Matrix**:
   - A `SymmetryMatrix` object `A` of size `3x3` is created.

3. **Print the Matrix**:
   - The `print` method is called to display the matrix.

4. **Catch Block**:
   - If an exception is thrown (e.g., invalid size), it is caught and an error message is printed.

#### **Why This Is Used**
- **Error Handling**: Ensures the program doesn’t crash if something goes wrong.
- **Demonstration**: Shows how to use the `SymmetryMatrix` class.

---

### **Summary**
This code is a well-structured implementation of a symmetric matrix class. It uses modern C++ features like vectors, random number generation, and mutexes to ensure correctness and thread safety. Each part of the code is designed to be clear, efficient, and easy to use.