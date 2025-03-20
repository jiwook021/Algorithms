# Step-by-Step Explanation: main.cpp

Let’s break down the code **step by step** in a way that’s accessible to everyone, from beginners to experts. I’ll explain each section in detail, define technical terms, and provide examples and diagrams where helpful.

---

### **1. Header Files and Includes**
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iomanip>
#include <string>
```

#### **What It Does**
These lines include external libraries that provide functionality for:
- **Input/output** (`iostream`): For printing to the console and reading user input.
- **Vectors** (`vector`): For storing collections of data points.
- **Mathematical functions** (`cmath`): For calculations like square roots.
- **Algorithms** (`algorithm`): For operations like sorting.
- **Random number generation** (`random`): For generating random data (not used in this snippet).
- **Exceptions** (`stdexcept`): For handling errors.
- **Formatting output** (`iomanip`): For controlling how numbers are displayed.
- **String manipulation** (`string`): For working with text.

#### **Why It’s Used**
These libraries are essential for the program to perform tasks like:
- Storing and manipulating data (`vector`).
- Performing mathematical operations (`cmath`).
- Handling errors gracefully (`stdexcept`).
- Interacting with the user (`iostream`).

---

### **2. DataPoint Struct**
```cpp
struct DataPoint {
    double x1;    // Feature 1
    double x2;    // Feature 2
    int label;    // Binary label (-1 or 1)

    // Constructor
    DataPoint(double x1, double x2, int label) : x1(x1), x2(x2), label(label) {}
};
```

#### **What It Does**
- Defines a `DataPoint` structure to represent a single data point.
- Each data point has:
  - Two features (`x1` and `x2`): Numerical values describing the data.
  - A label (`label`): A binary value (`-1` or `1`) indicating the class.

#### **Why It’s Used**
- **Structures** are used to group related data together.
- The constructor allows easy creation of `DataPoint` objects.

#### **Example**
```cpp
DataPoint dp(2.0, 3.0, -1);  // Creates a data point with x1=2.0, x2=3.0, label=-1
```

---

### **3. DataPreprocessor Class**
```cpp
class DataPreprocessor {
private:
    bool is_fitted = false;
    double x1_mean = 0.0, x2_mean = 0.0;
    double x1_std = 1.0, x2_std = 1.0;
```

#### **What It Does**
- Defines a class to preprocess data by normalizing it.
- Contains private member variables:
  - `is_fitted`: Tracks whether the preprocessor has been trained on data.
  - `x1_mean`, `x2_mean`: Store the mean (average) of each feature.
  - `x1_std`, `x2_std`: Store the standard deviation (spread) of each feature.

#### **Why It’s Used**
- Normalization ensures that all features are on the same scale, which is critical for machine learning algorithms like SVM.

---

### **4. fit() Method**
```cpp
void fit(const std::vector<DataPoint>& dataset) {
    if (dataset.empty()) {
        throw std::invalid_argument("Dataset is empty");
    }

    // Calculate means
    for (const auto& dp : dataset) {
        x1_mean += dp.x1;
        x2_mean += dp.x2;
    }
    x1_mean /= dataset.size();
    x2_mean /= dataset.size();

    // Calculate standard deviations
    for (const auto& dp : dataset) {
        x1_std += (dp.x1 - x1_mean) * (dp.x1 - x1_mean);
        x2_std += (dp.x2 - x2_mean) * (dp.x2 - x2_mean);
    }
    x1_std = std::sqrt(x1_std / dataset.size());
    x2_std = std::sqrt(x2_std / dataset.size());

    // Prevent division by zero
    x1_std = (x1_std < 1e-10) ? 1.0 : x1_std;
    x2_std = (x2_std < 1e-10) ? 1.0 : x2_std;

    is_fitted = true;
}
```

#### **What It Does**
1. **Checks for Empty Dataset**:
   - Throws an error if the dataset is empty.

2. **Calculates Means**:
   - Sums up all values of `x1` and `x2`.
   - Divides by the number of data points to get the average.

3. **Calculates Standard Deviations**:
   - Computes the squared difference between each value and the mean.
   - Takes the square root of the average squared differences.

4. **Prevents Division by Zero**:
   - If the standard deviation is too small, sets it to `1.0` to avoid errors.

5. **Marks as Fitted**:
   - Sets `is_fitted` to `true` to indicate the preprocessor is ready.

#### **Why It’s Used**
- **Mean and Standard Deviation** are used for Z-score normalization:
  - Formula: `normalized_value = (value - mean) / standard_deviation`
- This ensures the data has a mean of `0` and a standard deviation of `1`.

#### **Example**
For `x1 = [2, 3, 3, 5, 6, 7]`:
- Mean = `(2 + 3 + 3 + 5 + 6 + 7) / 6 = 4.33`
- Standard Deviation = `sqrt(((2-4.33)^2 + (3-4.33)^2 + ...) / 6) ≈ 1.86`

---

### **5. transform() Method**
```cpp
std::vector<DataPoint> transform(const std::vector<DataPoint>& dataset) const {
    if (!is_fitted) {
        throw std::runtime_error("Preprocessor must be fitted before transforming");
    }

    std::vector<DataPoint> normalized_data;
    normalized_data.reserve(dataset.size());

    for (const auto& dp : dataset) {
        double norm_x1 = (dp.x1 - x1_mean) / x1_std;
        double norm_x2 = (dp.x2 - x2_mean) / x2_std;
        normalized_data.emplace_back(norm_x1, norm_x2, dp.label);
    }

    return normalized_data;
}
```

#### **What It Does**
1. **Checks if Fitted**:
   - Throws an error if the preprocessor hasn’t been fitted.

2. **Normalizes Data**:
   - Applies the Z-score formula to each feature.
   - Stores the normalized data in a new vector.

3. **Returns Normalized Data**:
   - Returns the vector of normalized data points.

#### **Why It’s Used**
- Normalization ensures that all features contribute equally to the model.

#### **Example**
For `x1 = 2.0`, `x1_mean = 4.33`, `x1_std = 1.86`:
- Normalized `x1 = (2.0 - 4.33) / 1.86 ≈ -1.25`

---

### **6. Main Function**
```cpp
int main() {
    try {
        // Create dataset
        std::vector<DataPoint> dataset = {
            {2.0, 3.0, -1},  // Class -1
            {3.0, 3.0, -1},
            {3.0, 4.0, -1},
            {5.0, 5.0, 1},   // Class 1
            {6.0, 5.0, 1},
            {7.0, 6.0, 1}
        };

        // Preprocess data
        DataPreprocessor preprocessor;
        preprocessor.fit(dataset);
        auto normalized_dataset = preprocessor.transform(dataset);
        
        // Print preprocessing parameters
        preprocessor.print_parameters();

        // Select kernel type
        std::cout << "Select kernel type:" << std::endl;
        std::cout << "1. Linear" << std::endl;
        std::cout << "2. RBF (Gaussian)" << std::endl;
        std::cout << "Enter choice (1 or 2): ";
        
        int kernel_choice;
        std::cin >> kernel_choice;
        
        std::unique_ptr<Kernel> kernel;
        if (kernel_choice == 1) {
            kernel = std::make_uniq
```

#### **What It Does**
1. **Creates Dataset**:
   - Defines a dataset of 6 data points with features and labels.

2. **Preprocesses Data**:
   - Fits the preprocessor to the dataset.
   - Transforms the dataset into normalized form.

3. **Prints Parameters**:
   - Displays the mean and standard deviation used for normalization.

4. **Selects Kernel**:
   - Prompts the user to choose a kernel type for the SVM.

#### **Why It’s Used**
- Demonstrates the full workflow of preprocessing and kernel selection.

---

### **Summary**
This code is a well-structured implementation of data preprocessing for a machine learning system. It uses **Z-score normalization** to scale features and prepares the data for an SVM model. The modular design and error handling make it robust and easy to extend.