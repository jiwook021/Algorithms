# Step-by-Step Explanation: main.cpp

### Comprehensive, Step-by-Step Explanation of the Code

Let’s break down the code into its major sections and explain each part in detail. I’ll start from the top and work our way down, explaining every significant line of code, the logic behind it, and why it’s written the way it is.

---

### **1. Header Files and Imports**
```cpp
#include <iostream>
#include <vector>
#include <cmath>
```

#### What it does:
- These lines include necessary libraries for the program:
  - `<iostream>`: Provides input/output functionality (e.g., `std::cout` for printing to the console).
  - `<vector>`: Provides the `std::vector` container, which is used to store the dataset.
  - `<cmath>`: Provides mathematical functions like `std::max`.

#### Why it’s used:
- These libraries are essential for basic functionality like printing, storing data, and performing mathematical operations.

---

### **2. DataPoint Struct**
```cpp
struct DataPoint {
    double x1;  // Feature 1
    double x2;  // Feature 2
    int label;  // Binary label (-1 or 1 for SVM)
};
```

#### What it does:
- Defines a `struct` called `DataPoint` to represent a single data point in the dataset.
- Each `DataPoint` has:
  - `x1` and `x2`: Two features (numeric values) that describe the data point.
  - `label`: A binary label (`-1` or `1`) that indicates the class of the data point.

#### Why it’s used:
- A `struct` is a simple way to group related data together. Here, it’s used to store the features and label of each data point in a single object.

#### Example:
- A `DataPoint` could represent a point in 2D space, like `{2.0, 3.0, -1}`, where `x1 = 2.0`, `x2 = 3.0`, and the label is `-1`.

---

### **3. LinearSVM Class**
The `LinearSVM` class encapsulates the SVM model and its functionality.

#### **3.1 Private Members**
```cpp
private:
    double w1;  // Weight for feature x1
    double w2;  // Weight for feature x2
    double b;   // Bias term
    double learning_rate;
    double regularization;  // C parameter for regularization strength
    int epochs;
```

#### What it does:
- These are the internal variables of the SVM model:
  - `w1` and `w2`: Weights for the features `x1` and `x2`. These determine the slope of the decision boundary.
  - `b`: The bias term, which shifts the decision boundary up or down.
  - `learning_rate`: Controls how much the weights and bias are updated during training.
  - `regularization`: A parameter that controls the strength of regularization (to prevent overfitting).
  - `epochs`: The number of times the training algorithm will iterate over the dataset.

#### Why it’s used:
- These variables are essential for the SVM to learn the decision boundary. They are kept private to encapsulate the model’s internal state.

---

#### **3.2 Constructor**
```cpp
public:
    LinearSVM(double lr, double reg, int ep)
        : w1(0.0), w2(0.0), b(0.0), learning_rate(lr), regularization(reg), epochs(ep) {}
```

#### What it does:
- The constructor initializes the SVM model with:
  - `w1`, `w2`, and `b` set to `0.0` (initial weights and bias).
  - `learning_rate`, `regularization`, and `epochs` set to the values provided by the user.

#### Why it’s used:
- The constructor ensures that the model starts with a clean slate (weights and bias initialized to zero) and sets the hyperparameters for training.

---

#### **3.3 Decision Function**
```cpp
double decision_function(double x1, double x2) const {
    return w1 * x1 + w2 * x2 + b;
}
```

#### What it does:
- Computes the value of the decision function for a given data point `(x1, x2)`.
- The decision function is a linear combination of the features and weights, plus the bias:  
  `f(x) = w1 * x1 + w2 * x2 + b`.

#### Why it’s used:
- The decision function determines which side of the decision boundary a data point lies on. If `f(x) >= 0`, the point is classified as `1`; otherwise, it’s classified as `-1`.

#### Example:
- If `w1 = 1.0`, `w2 = -1.0`, `b = 0.5`, and the input is `(2.0, 3.0)`, then:  
  `f(x) = 1.0 * 2.0 + (-1.0) * 3.0 + 0.5 = -0.5`.  
  Since `-0.5 < 0`, the point is classified as `-1`.

---

#### **3.4 Predict Function**
```cpp
int predict(double x1, double x2) const {
    return decision_function(x1, x2) >= 0 ? 1 : -1;
}
```

#### What it does:
- Uses the decision function to classify a data point `(x1, x2)` into one of two classes (`1` or `-1`).

#### Why it’s used:
- This is the final step in the classification process. After training, the model uses this function to make predictions on new data.

---

#### **3.5 Hinge Loss**
```cpp
double hinge_loss(const DataPoint& dp) const {
    double margin = dp.label * decision_function(dp.x1, dp.x2);
    return std::max(0.0, 1.0 - margin);
}
```

#### What it does:
- Computes the hinge loss for a single data point.
- The hinge loss is defined as:  
  `max(0, 1 - y * f(x))`, where `y` is the true label and `f(x)` is the decision function.

#### Why it’s used:
- The hinge loss measures how well the model is classifying the data. It’s `0` if the point is correctly classified and outside the margin, and increases if the point is misclassified or within the margin.

#### Example:
- If `y = 1` and `f(x) = 0.8`, then:  
  `hinge_loss = max(0, 1 - 1 * 0.8) = 0.2`.  
  This indicates the point is close to the decision boundary.

---

#### **3.6 Train Function**
```cpp
void train(const std::vector<DataPoint>& dataset) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& dp : dataset) {
            double margin = dp.label * decision_function(dp.x1, dp.x2);
            if (margin < 1) {  // Point is misclassified or within the margin
                // Subgradient updates for weights and bias
                double grad_w1 = -dp.label * dp.x1 + (w1 / regularization);
                double grad_w2 = -dp.label * dp.x2 + (w2 / regularization);
                double grad_b = -dp.label;
                // Update parameters
                w1 -= learning_rate * grad_w1;
                w2 -= learning_rate * grad_w2;
                b -= learning_rate * grad_b;
            } else {  // Point is correctly classified outside the margin
                // Apply regularization to weights only
                w1 -= learning_rate * (w1 / regularization);
                w2 -= learning_rate * (w2 / regularization);
                // Bias is not regularized
            }
        }
    }
}
```

#### What it does:
- Trains the SVM model using gradient descent.
- For each epoch (iteration over the dataset), it:
  1. Computes the margin for each data point.
  2. If the margin is less than `1` (misclassified or within the margin), it updates the weights and bias using the hinge loss subgradient.
  3. If the margin is greater than or equal to `1` (correctly classified), it applies regularization to the weights.

#### Why it’s used:
- Gradient descent is used to minimize the hinge loss and find the optimal weights and bias. Regularization prevents overfitting by penalizing large weights.

#### Example:
- Suppose `w1 = 0.5`, `w2 = -0.5`, `b = 0.0`, and the data point is `{2.0, 3.0, -1}`. The margin is:  
  `margin = -1 * (0.5 * 2.0 + (-0.5) * 3.0 + 0.0) = -1 * (1.0 - 1.5) = 0.5`.  
  Since `0.5 < 1`, the weights and bias are updated.

---

#### **3.7 Print Parameters**
```cpp
void print_parameters() const {
    std::cout << "Learned weights: w1 = " << w1 << ", w2 = " << w2 << ", b = " << b << std::endl;
}
```

#### What it does:
- Prints the learned weights and bias to the console.

#### Why it’s used:
- This is useful for debugging and understanding what the model has learned.

---

### **4. Main Function**
```cpp
int main() {
    // Hardcoded dataset: {x1, x2, label}
    std::vector<DataPoint> dataset = {
        {2.0, 3.0, -1},  // Class -1
        {3.0, 3.0, -1},
        {3.0, 4.0, -1},
        {5.0, 5.0, 1},   // Class 1
        {6.0, 5.0, 1},
        {7.0, 6.0, 1}
    };

    // Initialize SVM with learning rate = 0.01, regularization = 1.0, 100 epochs
    LinearSVM model(0.01, 1.0, 100);
    model.train(dataset);

    // Show the learned parameters
    model.print_parameters();

    // User input for prediction
    std::cout << "Enter x1 and x2 (e.g., 4.0 4.0): ";
    double x1, x2;
    std::cin >> x1 >> x2;

    // Predict and display the result
    int prediction = model.predict(x1, x2);
    std::cout << "Predicted class: " << prediction << std::endl;

    return 0;
}
```

#### What it does:
1. Creates a hardcoded dataset of `DataPoint` objects.
2. Initializes an `LinearSVM` object with specific hyperparameters.
3. Trains the model on the dataset.
4. Prints the learned weights and bias.
5. Allows the user to input new data points and predicts their class.

#### Why it’s used:
- This is the entry point of the program. It ties everything together and demonstrates how the SVM works in practice.

---

### **Summary of Control Flow**
1. The program starts by defining the dataset.
2. It initializes the SVM model with hyperparameters.
3. The model is trained on the dataset using gradient descent.
4. After training, the learned parameters are displayed.
5. The user can input new data points, and the model predicts their class.

This code is a complete implementation of a linear SVM for binary classification, with clear separation of data, model, and user interaction. It’s designed to be easy to understand and extend.