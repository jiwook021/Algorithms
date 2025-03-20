# Code Overview: main.cpp

This C++ code implements a **Logistic Regression** model with advanced features for binary classification. Let's break down its purpose, functionality, and structure in detail:

---

### **1. Problem Being Solved**
The code is designed to solve a **binary classification problem**, where the goal is to predict one of two possible outcomes (e.g., pass/fail, yes/no, 0/1). In this specific case:
- The input data consists of two main features:
  - `x1`: Represents a feature like "study hours."
  - `x2`: Represents a feature like "IQ points above baseline."
- The output is a binary label (`0` or `1`), where:
  - `0` might represent "fail."
  - `1` might represent "pass."

The model uses **Logistic Regression**, a machine learning algorithm, to predict the probability of the positive class (`1`) based on the input features.

---

### **2. Main Functionality**
The code performs the following key tasks:

#### **a. Data Representation**
- The `DataPoint` struct represents a single data point with:
  - Two main features (`x1` and `x2`).
  - Three engineered features (`x3`, `x4`, `x5`):
    - `x3`: Interaction term (`x1 * x2`).
    - `x4`: Polynomial feature (`x1^2`).
    - `x5`: Polynomial feature (`x2^2`).
  - A binary `label` (0 or 1).

#### **b. Feature Engineering**
- The `add_engineered_features` function adds interaction and polynomial features to the dataset. These features help capture more complex relationships between the input variables, improving the model's predictive power.

#### **c. Feature Normalization**
- The `normalize_features` function scales all features to a range of `[0, 1]` using min-max normalization. This ensures that all features contribute equally to the model, regardless of their original scales.

#### **d. Logistic Regression Model**
- The `LogisticRegression` class implements the logistic regression algorithm with advanced features:
  - **Sigmoid Function**: Maps the model's output to a probability between 0 and 1.
  - **Binary Cross-Entropy Loss**: Measures the difference between the predicted and actual labels.
  - **Gradient Descent**: Optimizes the model's weights and bias to minimize the loss.
  - **Advanced Features**:
    - **L2 Regularization**: Prevents overfitting by penalizing large weights.
    - **Learning Rate Decay**: Reduces the learning rate over time for better convergence.
    - **Momentum**: Accelerates gradient descent by considering past updates.
    - **Early Stopping**: Stops training if the validation loss doesn't improve for a specified number of epochs.

#### **e. Model Evaluation**
- The `Metrics` struct stores evaluation metrics like accuracy, precision, recall, F1 score, and AUC (Area Under the ROC Curve). These metrics help assess the model's performance.

---

### **3. Algorithms Used**
The code uses the following algorithms and techniques:

#### **a. Logistic Regression**
- Logistic Regression is a statistical model used for binary classification. It predicts the probability of the positive class using the sigmoid function:
  \[
  P(y=1) = \frac{1}{1 + e^{-(w \cdot x + b)}}
  \]
  where:
  - \( w \): Weights for the features.
  - \( x \): Input features.
  - \( b \): Bias term.

#### **b. Gradient Descent**
- Gradient Descent is an optimization algorithm used to minimize the loss function. It updates the model's weights and bias iteratively:
  \[
  w = w - \eta \cdot \frac{\partial L}{\partial w}
  \]
  where:
  - \( \eta \): Learning rate.
  - \( L \): Loss function.

#### **c. Feature Engineering**
- Interaction and polynomial features are added to capture non-linear relationships in the data.

#### **d. Regularization**
- L2 regularization is used to prevent overfitting by adding a penalty term to the loss function:
  \[
  L_{\text{regularized}} = L + \frac{\lambda}{2} \sum w^2
  \]
  where \( \lambda \) is the regularization strength.

#### **e. Early Stopping**
- Training stops early if the validation loss doesn't improve for a specified number of epochs (`patience`), preventing overfitting.

---

### **4. Overall Structure**
The code is organized into the following components:

#### **a. Data Structures**
- `DataPoint`: Represents a single data point with features and a label.
- `Metrics`: Stores evaluation metrics.

#### **b. Functions**
- `add_engineered_features`: Adds interaction and polynomial features.
- `normalize_features`: Normalizes features to the range `[0, 1]`.
- `normalize_datapoint`: Normalizes a single data point.
- `sigmoid`: Computes the sigmoid function.
- `binary_cross_entropy`: Computes the binary cross-entropy loss.

#### **c. Classes**
- `LogisticRegression`: Implements the logistic regression model with advanced features like momentum, learning rate decay, and early stopping.

---

### **5. How the Parts Work Together**
1. **Data Preparation**:
   - Data points are created with two main features (`x1` and `x2`).
   - Interaction and polynomial features are added using `add_engineered_features`.
   - Features are normalized using `normalize_features`.

2. **Model Training**:
   - The `LogisticRegression` class is initialized with hyperparameters like learning rate, epochs, and regularization strength.
   - The model is trained using gradient descent, with optional momentum and learning rate decay.
   - Early stopping is used to prevent overfitting.

3. **Model Evaluation**:
   - The model's performance is evaluated using metrics like accuracy, precision, recall, F1 score, and AUC.

4. **Prediction**:
   - The trained model can predict the probability of the positive class for new data points.

---

### **6. Example Use Case**
Imagine a dataset where:
- `x1` represents the number of hours a student studies.
- `x2` represents their IQ score above a baseline.
- The label indicates whether the student passed (`1`) or failed (`0`) an exam.

The code would:
1. Add interaction and polynomial features to capture complex relationships (e.g., studying more hours might have a stronger effect for students with higher IQs).
2. Normalize the features to ensure equal contribution.
3. Train a logistic regression model to predict the probability of passing based on the input features.
4. Evaluate the model's performance using metrics like accuracy and AUC.

---

### **7. Key Takeaways**
- The code implements a **robust logistic regression model** with advanced features like feature engineering, normalization, regularization, and early stopping.
- It is designed to handle **binary classification problems** where the goal is to predict one of two possible outcomes.
- The modular structure makes it easy to extend or modify for different datasets or use cases.

This explanation should give you a solid understanding of the code's purpose and functionality. Let me know if you'd like to dive deeper into any specific part!