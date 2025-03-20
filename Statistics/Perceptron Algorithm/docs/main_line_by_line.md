# Step-by-Step Explanation: main.cpp

Let’s dive into a **comprehensive, step-by-step explanation** of the code. I’ll break it down into sections, explain each part in simple terms, and provide examples and diagrams where necessary. By the end, you’ll understand not only **what** the code does but also **why** it works the way it does.

---

### **1. Header Files and Includes**
```cpp
#include <iostream>
#include <vector>
#include <cstdlib>  // For rand()
#include <ctime>    // For time()
```

#### **What It Does**
These lines include libraries that provide functionality for:
- **Input/Output** (`<iostream>`): For printing to the console and reading user input.
- **Vectors** (`<vector>`): For storing collections of data (like the dataset).
- **Random Number Generation** (`<cstdlib>`): For generating random weights.
- **Time Functions** (`<ctime>`): For seeding the random number generator.

#### **Why It’s Used**
- `iostream` is necessary for interacting with the user (e.g., printing results and taking input).
- `vector` is used to store the dataset because it’s a flexible and efficient way to handle collections of data.
- `cstdlib` and `ctime` are used to initialize the Perceptron’s weights randomly, which is a common practice in machine learning.

---

### **2. DataPoint Struct**
```cpp
struct DataPoint {
    double x1;  // Feature 1 (e.g., study hours)
    double x2;  // Feature 2 (e.g., sleep hours)
    int label;  // Binary label (0 or 1)
};
```

#### **What It Does**
This defines a **structure** (a custom data type) called `DataPoint`. Each `DataPoint` represents a single data point in the dataset and contains:
- `x1`: The first feature (e.g., study hours).
- `x2`: The second feature (e.g., sleep hours).
- `label`: The class label (e.g., `0` for fail, `1` for pass).

#### **Why It’s Used**
- A `struct` is used to group related data together. This makes it easy to store and manipulate the dataset.
- The `label` is necessary for supervised learning because the Perceptron needs to know the correct answer during training.

#### **Example**
```cpp
DataPoint dp = {2.0, 6.0, 0};  // A student who studied 2 hours, slept 6 hours, and failed
```

---

### **3. Perceptron Class**
The `Perceptron` class encapsulates the logic for training and prediction. Let’s break it down.

#### **3.1 Private Members**
```cpp
private:
    double w1;  // Weight for x1
    double w2;  // Weight for x2
    double b;   // Bias
    double learning_rate;
    int epochs;
```

#### **What It Does**
These are the **private member variables** of the `Perceptron` class:
- `w1` and `w2`: Weights for the input features `x1` and `x2`.
- `b`: Bias term, which shifts the decision boundary.
- `learning_rate`: Controls how much the weights and bias are adjusted during training.
- `epochs`: The number of times the Perceptron iterates over the dataset during training.

#### **Why It’s Used**
- The weights and bias are the parameters the Perceptron learns. They define the decision boundary.
- The `learning_rate` ensures that the Perceptron doesn’t overcorrect its weights, which could lead to instability.
- `epochs` determine how many times the Perceptron sees the dataset during training.

---

#### **3.2 Constructor**
```cpp
Perceptron(double lr, int ep) : learning_rate(lr), epochs(ep) {
    std::srand(std::time(0));  // Seed for random weights
    w1 = (std::rand() % 1000) / 1000.0;  // Random weight between 0 and 1
    w2 = (std::rand() % 1000) / 1000.0;
    b = (std::rand() % 1000) / 1000.0;
}
```

#### **What It Does**
- The constructor initializes the Perceptron with a learning rate (`lr`) and number of epochs (`ep`).
- It seeds the random number generator using the current time (`std::time(0)`).
- It initializes `w1`, `w2`, and `b` with random values between `0` and `1`.

#### **Why It’s Used**
- Random initialization ensures that the Perceptron starts with different weights each time, which helps avoid getting stuck in a local minimum.
- Seeding with the current time ensures that the random numbers are different each time the program runs.

---

#### **3.3 Step Function**
```cpp
int step_function(double sum) const {
    return (sum >= 0) ? 1 : 0;
}
```

#### **What It Does**
- This is the **activation function**. It takes a weighted sum (`sum`) and returns `1` if the sum is greater than or equal to `0`, and `0` otherwise.

#### **Why It’s Used**
- The step function is used to make binary decisions. It’s simple and works well for linearly separable data.

#### **Example**
```cpp
double sum = 0.5;  // Weighted sum
int prediction = step_function(sum);  // prediction = 1
```

---

#### **3.4 Train Method**
```cpp
void train(const std::vector<DataPoint>& dataset) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& dp : dataset) {
            // Compute weighted sum
            double sum = w1 * dp.x1 + w2 * dp.x2 + b;
            int prediction = step_function(sum);
            int error = dp.label - prediction;

            // Update weights and bias if there's an error
            if (error != 0) {
                w1 += learning_rate * error * dp.x1;
                w2 += learning_rate * error * dp.x2;
                b += learning_rate * error;
            }
        }
    }
}
```

#### **What It Does**
- The `train` method iterates over the dataset for the specified number of epochs.
- For each data point, it:
  1. Computes the weighted sum: `sum = w1 * x1 + w2 * x2 + b`.
  2. Predicts the class using the step function.
  3. Calculates the error: `error = true label - predicted label`.
  4. Updates the weights and bias if there’s an error.

#### **Why It’s Used**
- This is the core of the Perceptron learning algorithm. It adjusts the weights and bias to minimize classification errors.
- The updates are proportional to the error and the input features, scaled by the learning rate.

#### **Example**
Suppose:
- `w1 = 0.5`, `w2 = 0.3`, `b = -1.0`
- `x1 = 2.0`, `x2 = 6.0`, `label = 0`
- `learning_rate = 0.1`

1. Compute sum: `sum = 0.5 * 2.0 + 0.3 * 6.0 + (-1.0) = 1.8`
2. Predict: `prediction = step_function(1.8) = 1`
3. Error: `error = 0 - 1 = -1`
4. Update weights:
   - `w1 += 0.1 * (-1) * 2.0 = 0.5 - 0.2 = 0.3`
   - `w2 += 0.1 * (-1) * 6.0 = 0.3 - 0.6 = -0.3`
   - `b += 0.1 * (-1) = -1.0 - 0.1 = -1.1`

---

#### **3.5 Predict Method**
```cpp
int predict(double x1, double x2) const {
    double sum = w1 * x1 + w2 * x2 + b;
    return step_function(sum);
}
```

#### **What It Does**
- This method predicts the class of a new data point by computing the weighted sum and applying the step function.

#### **Why It’s Used**
- After training, the Perceptron can make predictions on new, unseen data.

#### **Example**
```cpp
int prediction = model.predict(5.0, 4.0);  // Predicts whether a student who studied 5 hours and slept 4 hours will pass
```

---

#### **3.6 Print Parameters Method**
```cpp
void print_parameters() const {
    std::cout << "Learned weights: w1 = " << w1 << ", w2 = " << w2 << ", b = " << b << std::endl;
}
```

#### **What It Does**
- This method prints the learned weights and bias.

#### **Why It’s Used**
- It’s useful for debugging and understanding what the Perceptron has learned.

---

### **4. Main Function**
The `main` function ties everything together.

#### **4.1 Dataset Creation**
```cpp
std::vector<DataPoint> dataset = {
    {2.0, 6.0, 0},  // Fail
    {4.0, 5.0, 0},  // Fail
    {3.0, 7.0, 0},  // Fail
    {5.0, 4.0, 1},  // Pass
    {6.0, 6.0, 1},  // Pass
    {7.0, 5.0, 1}   // Pass
};
```

#### **What It Does**
- This creates a dataset of `DataPoint` objects, each representing a student’s study hours, sleep hours, and whether they passed or failed.

#### **Why It’s Used**
- The dataset is used to train the Perceptron.

---

#### **4.2 Perceptron Initialization and Training**
```cpp
Perceptron model(0.1, 10);  // Learning rate = 0.1, 10 epochs
model.train(dataset);
```

#### **What It Does**
- Initializes a `Perceptron` object with a learning rate of `0.1` and `10` epochs.
- Trains the Perceptron on the dataset.

#### **Why It’s Used**
- Training allows the Perceptron to learn the optimal weights and bias for classification.

---

#### **4.3 User Input and Prediction**
```cpp
std::cout << "Enter study hours and sleep hours (e.g., 5.0 6.0): ";
double x1, x2;
std::cin >> x1 >> x2;

int prediction = model.predict(x1, x2);
std::cout << "Predicted class (0 = fail, 1 = pass): " << prediction << std::endl;
```

#### **What It Does**
- Prompts the user to input study and sleep hours.
- Predicts whether the student will pass or fail.

#### **Why It’s Used**
- This demonstrates the Perceptron’s ability to make predictions on new data.

---

### **5. Summary**
This code implements a Perceptron for binary classification. It:
1. Defines a dataset of students’ study and sleep hours.
2. Trains the Perceptron to learn a decision boundary.
3. Allows the user to input new data and predicts the outcome.

By breaking down each part of the code, we’ve seen how the Perceptron works, why certain techniques are used, and how everything fits together. This is a great starting point for understanding machine learning algorithms!