# Code Overview: main.cpp

This C++ code implements a **Logistic Regression** model, which is a fundamental machine learning algorithm used for **binary classification** tasks. The purpose of this code is to predict whether a student will **pass (1)** or **fail (0)** an exam based on two features:
1. **Study hours (x1)**: The number of hours a student has studied.
2. **IQ points above baseline (x2)**: The student's IQ score relative to a baseline IQ of 80.

The code is structured to preprocess the data, train the logistic regression model, and evaluate its performance. Below is a detailed explanation of the main functionality, algorithms, and structure:

---

### **1. Problem Being Solved**
The problem is a **binary classification task**, where the goal is to predict whether a student will pass or fail based on their study hours and IQ. The dataset consists of labeled examples, where each example has:
- Two features: `x1` (study hours) and `x2` (IQ points above baseline).
- A binary label: `0` (fail) or `1` (pass).

The logistic regression model learns the relationship between the features and the label, allowing it to make predictions on new, unseen data.

---

### **2. Approach Taken**
The code follows these steps to solve the problem:

#### **a. Data Representation**
- The data is represented using a `DataPoint` structure, which stores:
  - `x1`: Study hours (feature 1).
  - `x2`: IQ points above baseline (feature 2).
  - `label`: The binary label (0 or 1).

#### **b. Data Preprocessing**
- The features (`x1` and `x2`) are **normalized** to the range `[0, 1]` using the `normalize_features` function. Normalization ensures that both features contribute equally to the model's learning process, regardless of their original scales.
- The normalization process involves:
  1. Finding the minimum and maximum values for each feature.
  2. Scaling each feature to the range `[0, 1]` using the formula:
     \[
     \text{normalized\_value} = \frac{\text{original\_value} - \text{min}}{\text{max} - \text{min}}
     \]

#### **c. Logistic Regression Model**
- The `LogisticRegression` class implements the logistic regression algorithm, which is a supervised learning algorithm for binary classification.
- The model learns the relationship between the features and the label by optimizing a set of parameters:
  - `w1`: Weight for feature 1 (study hours).
  - `w2`: Weight for feature 2 (IQ points above baseline).
  - `b`: Bias term.
- The model uses the **sigmoid function** to map the linear combination of features and weights to a probability between 0 and 1:
  \[
  \text{sigmoid}(z) = \frac{1}{1 + e^{-z}}
  \]
  where \( z = w1 \cdot x1 + w2 \cdot x2 + b \).

#### **d. Loss Function**
- The model uses the **binary cross-entropy loss** function to measure how well the predicted probabilities match the true labels. The loss function is defined as:
  \[
  \text{loss} = -\left( y_{\text{true}} \cdot \log(y_{\text{pred}}) + (1 - y_{\text{true}}) \cdot \log(1 - y_{\text{pred}}) \right)
  \]
  where:
  - \( y_{\text{true}} \) is the true label (0 or 1).
  - \( y_{\text{pred}} \) is the predicted probability (output of the sigmoid function).

#### **e. Training**
- The model is trained using **gradient descent**, an optimization algorithm that iteratively updates the weights and bias to minimize the loss function.
- The training process includes:
  - **Early stopping**: To prevent overfitting, the training stops early if the validation loss does not improve for a specified number of epochs (`patience`).
  - **L2 regularization**: A penalty term is added to the loss function to prevent the weights from growing too large, which helps improve generalization.

#### **f. Evaluation**
- The model's performance is evaluated using the validation set, and the training and validation losses are tracked over time to monitor the learning process.

---

### **3. Overall Structure**
The code is organized into the following components:

1. **Data Representation**:
   - The `DataPoint` structure defines how each data point is stored.

2. **Data Preprocessing**:
   - The `normalize_features` function scales the features to the range `[0, 1]`.

3. **Model Definition**:
   - The `LogisticRegression` class encapsulates the logistic regression model, including:
     - Weights (`w1`, `w2`) and bias (`b`).
     - Training parameters (learning rate, max epochs, etc.).
     - Methods for training and evaluation.

4. **Utility Functions**:
   - `sigmoid`: Computes the sigmoid function.
   - `binary_cross_entropy`: Computes the loss function.

5. **Main Function**:
   - The `main` function initializes the dataset, preprocesses the data, and trains the model.

---

### **4. Algorithms Used**
- **Logistic Regression**: A linear model for binary classification.
- **Gradient Descent**: An optimization algorithm for minimizing the loss function.
- **Sigmoid Function**: Maps the linear combination of features and weights to a probability.
- **Binary Cross-Entropy Loss**: Measures the difference between predicted and true labels.
- **L2 Regularization**: Adds a penalty term to the loss function to prevent overfitting.
- **Early Stopping**: Stops training if the validation loss does not improve for a specified number of epochs.

---

### **5. How the Parts Work Together**
1. The `main` function initializes the dataset and preprocesses it using `normalize_features`.
2. The `LogisticRegression` class is instantiated with training parameters.
3. The model is trained using gradient descent, with early stopping and L2 regularization.
4. The training and validation losses are tracked to monitor the model's performance.
5. Once trained, the model can make predictions on new data by computing the sigmoid of the linear combination of features and weights.

---

### **Summary**
This code implements a logistic regression model to predict whether a student will pass or fail based on their study hours and IQ. It includes data preprocessing, model training, and evaluation, with features like early stopping and L2 regularization to improve performance and prevent overfitting. The code is well-structured and modular, making it easy to extend or modify for other binary classification tasks.