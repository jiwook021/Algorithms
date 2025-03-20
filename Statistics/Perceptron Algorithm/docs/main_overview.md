# Code Overview: main.cpp

This C++ code implements a **Perceptron**, which is one of the simplest types of **artificial neural networks**. The Perceptron is used for **binary classification**, meaning it can classify data into one of two categories (in this case, labeled as `0` or `1`). Let’s break down the purpose, functionality, and structure of the code in detail.

---

### **Problem Being Solved**
The code is designed to solve a **binary classification problem**. Specifically, it predicts whether a student will **pass** (`1`) or **fail** (`0`) an exam based on two features:
1. **Study hours** (`x1`): The number of hours the student studied.
2. **Sleep hours** (`x2`): The number of hours the student slept.

The goal is to train the Perceptron to learn a decision boundary that separates the two classes (pass/fail) based on the provided dataset.

---

### **Approach Taken**
The Perceptron algorithm is a **supervised learning** algorithm. It learns from labeled data (the dataset) by adjusting its internal parameters (weights and bias) to minimize classification errors. Here’s how it works:

1. **Input Features**: The Perceptron takes two input features (`x1` and `x2`) and combines them with learned weights (`w1` and `w2`) and a bias (`b`) to compute a weighted sum.
2. **Activation Function**: The weighted sum is passed through a **step function**, which outputs `1` if the sum is greater than or equal to `0`, and `0` otherwise.
3. **Training**: During training, the Perceptron adjusts its weights and bias based on the error between its prediction and the true label. This process is repeated for multiple **epochs** (iterations over the dataset).
4. **Prediction**: Once trained, the Perceptron can predict the class of new, unseen data points.

---

### **Overall Structure**
The code is organized into three main parts:
1. **Data Representation**: The `DataPoint` struct represents a single data point with two features (`x1`, `x2`) and a label (`label`).
2. **Perceptron Class**: The `Perceptron` class encapsulates the logic for training and prediction. It includes:
   - **Weights and Bias**: These are the parameters the Perceptron learns.
   - **Step Function**: A simple activation function for binary classification.
   - **Training Method**: Adjusts the weights and bias based on errors.
   - **Prediction Method**: Predicts the class of new data points.
3. **Main Function**: This is the entry point of the program. It:
   - Creates a dataset of labeled examples.
   - Initializes and trains the Perceptron.
   - Allows the user to input new data points for prediction.

---

### **How the Code Works Together**
1. **Dataset Creation**: The `main` function defines a hardcoded dataset of `DataPoint` objects. Each object represents a student’s study hours, sleep hours, and whether they passed (`1`) or failed (`0`).
2. **Perceptron Initialization**: A `Perceptron` object is created with a learning rate (`0.1`) and number of epochs (`10`). The weights and bias are initialized randomly.
3. **Training**: The `train` method iterates over the dataset for the specified number of epochs. For each data point, it:
   - Computes the weighted sum.
   - Predicts the class using the step function.
   - Adjusts the weights and bias if the prediction is incorrect.
4. **Prediction**: After training, the user can input new values for study hours and sleep hours. The Perceptron predicts whether the student will pass or fail based on the learned weights and bias.
5. **Output**: The learned weights and bias are displayed, and the prediction result is shown to the user.

---

### **Algorithms Used**
1. **Perceptron Learning Algorithm**:
   - Computes the weighted sum: `sum = w1 * x1 + w2 * x2 + b`.
   - Applies the step function to classify the data point.
   - Updates weights and bias using the formula:
     ```
     w1 += learning_rate * error * x1
     w2 += learning_rate * error * x2
     b += learning_rate * error
     ```
   - The `error` is the difference between the true label and the predicted label.

2. **Step Function**:
   - A simple activation function that outputs `1` if the weighted sum is non-negative, and `0` otherwise.

---

### **Key Concepts**
1. **Weights and Bias**: These are the parameters the Perceptron learns. They define the decision boundary that separates the two classes.
2. **Learning Rate**: Controls how much the weights and bias are adjusted during training. A smaller learning rate leads to slower but more stable learning.
3. **Epochs**: The number of times the Perceptron iterates over the entire dataset during training.
4. **Binary Classification**: The Perceptron is designed for problems where the output is one of two classes (e.g., pass/fail, spam/not spam).

---

### **Example Walkthrough**
1. **Dataset**:
   - A student who studied for `2` hours and slept for `6` hours failed (`0`).
   - A student who studied for `6` hours and slept for `6` hours passed (`1`).

2. **Training**:
   - The Perceptron starts with random weights and bias.
   - It iterates over the dataset, adjusting the weights and bias to minimize errors.

3. **Prediction**:
   - After training, the Perceptron can predict whether a new student will pass or fail based on their study and sleep hours.

---

### **Summary**
This code demonstrates how a Perceptron can be implemented in C++ to solve a simple binary classification problem. It uses a step function for activation, adjusts weights and bias during training, and can make predictions on new data. The code is structured to be modular, with clear separation between data representation, the Perceptron logic, and the main program flow. This makes it easy to understand, extend, and modify for other classification tasks.