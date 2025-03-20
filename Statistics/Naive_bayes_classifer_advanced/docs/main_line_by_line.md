# Step-by-Step Explanation: main.cpp

Let’s dive into the code step by step, breaking it down into digestible parts. I’ll explain each section in detail, define technical terms, and provide examples where necessary. We’ll start with the **includes** and then move through each class and method.

---

### **1. Includes**
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <iomanip>
#include <limits>
```

#### **What It Does**
These are **header files** that provide functionality for:
- **Input/Output** (`<iostream>`): For reading from and writing to the console.
- **Math Operations** (`<cmath>`): For mathematical functions like `sqrt` and `pow`.
- **Data Structures** (`<vector>`, `<unordered_map>`): For storing collections of data.
- **Optional Values** (`<optional>`): For representing values that may or may not exist.
- **Error Handling** (`<stdexcept>`): For throwing and catching exceptions.
- **Memory Management** (`<memory>`): For smart pointers like `std::unique_ptr`.
- **Formatting** (`<iomanip>`): For formatting output (e.g., setting decimal precision).

#### **Why It’s Used**
These headers are included to provide the necessary tools for the program to work. For example:
- `<cmath>` is needed for calculating the Gaussian probability density.
- `<optional>` is used to represent a class label that might not exist (e.g., for unlabeled data points).

---

### **2. DataPoint Class**
```cpp
class DataPoint {
public:
    DataPoint(double x1, double x2, std::optional<int> label = std::nullopt)
        : x1_(x1), x2_(x2), label_(label) {}
```

#### **What It Does**
This class represents a **data point** with two features (`x1` and `x2`) and an optional **class label** (`label_`).

#### **Key Concepts**
- **Class**: A blueprint for creating objects. Here, `DataPoint` is a class that represents a single data point.
- **Constructor**: A special method that initializes an object. The constructor here takes `x1`, `x2`, and an optional `label`.
- **Optional**: A type that can either hold a value or be empty (`std::nullopt`). This is useful for representing data points that might not have a label.

#### **Example**
```cpp
DataPoint point1(3.5, 4.5, 1);  // A labeled data point
DataPoint point2(2.0, 5.0);     // An unlabeled data point
```

---

### **3. Getters and Setters**
```cpp
double getX1() const { return x1_; }
double getX2() const { return x2_; }
std::optional<int> getLabel() const { return label_; }

void setLabel(int label) { 
    if (label != 0 && label != 1) {
        throw std::invalid_argument("Label must be 0 or 1");
    }
    label_ = label; 
}
```

#### **What It Does**
- **Getters**: Methods to retrieve the values of `x1`, `x2`, and `label_`.
- **Setter**: A method to set the `label_`, with validation to ensure it’s either 0 or 1.

#### **Why It’s Used**
- **Encapsulation**: Getters and setters control access to private data, ensuring it’s only modified in valid ways.
- **Validation**: The `setLabel` method ensures the label is always 0 or 1, preventing invalid data.

#### **Example**
```cpp
DataPoint point(1.0, 2.0);
point.setLabel(1);  // Valid
point.setLabel(2);  // Throws an exception
```

---

### **4. fromUserInput Method**
```cpp
static DataPoint fromUserInput() {
    double x1, x2;
    std::cout << "Enter x1 and x2 features (e.g., 3.5 4.5): ";
    if (!(std::cin >> x1 >> x2)) {
        std::cin.clear();  // Clear error state
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');  // Discard invalid input
        throw std::runtime_error("Invalid input format. Please enter two numeric values.");
    }
    return DataPoint(x1, x2);
}
```

#### **What It Does**
This method reads two numbers (`x1` and `x2`) from the user and creates a `DataPoint` object.

#### **Key Concepts**
- **Static Method**: A method that belongs to the class rather than an instance. It can be called without creating an object.
- **Input Validation**: Checks if the input is valid (e.g., numeric). If not, it clears the error state and throws an exception.

#### **Why It’s Used**
- **User Interaction**: Allows the program to interact with the user and create data points dynamically.
- **Error Handling**: Ensures the program doesn’t crash if the user enters invalid input.

#### **Example**
```cpp
DataPoint point = DataPoint::fromUserInput();
// If the user enters "3.5 4.5", point will have x1=3.5 and x2=4.5.
```

---

### **5. Distribution Interface**
```cpp
class Distribution {
public:
    virtual ~Distribution() = default;
    virtual double probabilityDensity(double x) const = 0;
    virtual double logProbabilityDensity(double x) const = 0;
    virtual std::unique_ptr<Distribution> clone() const = 0;
};
```

#### **What It Does**
This is an **abstract base class** that defines an interface for probability distributions.

#### **Key Concepts**
- **Abstract Class**: A class that cannot be instantiated directly. It’s meant to be inherited by other classes.
- **Pure Virtual Functions**: Functions marked with `= 0` that must be implemented by derived classes.

#### **Why It’s Used**
- **Polymorphism**: Allows different types of distributions (e.g., Gaussian) to be used interchangeably.
- **Encapsulation**: Hides the implementation details of specific distributions.

---

### **6. GaussianDistribution Class**
```cpp
class GaussianDistribution : public Distribution {
public:
    GaussianDistribution(double mean, double variance)
        : mean_(mean), variance_(variance) {
        if (variance <= 0.0) {
            throw std::invalid_argument("Variance must be positive");
        }
        stdDev_ = std::sqrt(variance);
        constexpr double PI = 3.141592653589793;
        normalizationFactor_ = 1.0 / (stdDev_ * std::sqrt(2.0 * PI));
    }
```

#### **What It Does**
This class implements a **Gaussian (Normal) distribution**, which is a common probability distribution used in statistics.

#### **Key Concepts**
- **Mean**: The average value of the distribution.
- **Variance**: A measure of how spread out the values are.
- **Standard Deviation**: The square root of the variance.

#### **Why It’s Used**
- **Modeling Data**: The Gaussian distribution is often used to model real-world data because many natural phenomena follow this distribution.
- **Efficiency**: The normalization factor is pre-calculated to avoid redundant computations.

#### **Example**
```cpp
GaussianDistribution dist(0.0, 1.0);  // Mean = 0, Variance = 1
double prob = dist.probabilityDensity(1.0);  // Probability density at x=1.0
```

---

### **7. probabilityDensity Method**
```cpp
double probabilityDensity(double x) const override {
    double exponent = -std::pow(x - mean_, 2) / (2.0 * variance_);
    if (exponent < -700.0) return 0.0;
    return normalizationFactor_ * std::exp(exponent);
}
```

#### **What It Does**
This method calculates the **probability density** of a value `x` under the Gaussian distribution.

#### **Key Concepts**
- **Probability Density**: The likelihood of a value occurring in a distribution.
- **Exponent**: The part of the Gaussian formula that determines how quickly the probability decreases as `x` moves away from the mean.

#### **Why It’s Used**
- **Numerical Stability**: The check `if (exponent < -700.0)` prevents underflow (extremely small numbers that can’t be represented accurately).

#### **Example**
For a Gaussian with mean=0 and variance=1:
- `probabilityDensity(0.0)` returns the highest value (the peak of the bell curve).
- `probabilityDensity(10.0)` returns a very small value (far from the mean).

---

### **8. FeatureStats Class**
```cpp
class FeatureStats {
public:
    FeatureStats() : mean_(0.0), variance_(0.0), count_(0) {}
```

#### **What It Does**
This class calculates and stores statistics (mean, variance, etc.) for a feature across data points of the same class.

#### **Key Concepts**
- **Mean**: The average value of the feature.
- **Variance**: A measure of how spread out the feature values are.
- **Count**: The number of data points used to calculate the statistics.

#### **Why It’s Used**
- **Model Training**: These statistics are used to estimate the parameters of the Gaussian distributions for each feature and class.

---

### **Summary**
This code implements a Naive Bayes classifier using Gaussian distributions to model feature probabilities. It’s designed with modularity, numerical stability, and error handling in mind. Each class has a clear responsibility, and the code is structured to make it easy to extend and reuse.