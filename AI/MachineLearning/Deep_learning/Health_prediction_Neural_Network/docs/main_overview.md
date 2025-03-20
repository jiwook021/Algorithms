# Code Overview: main.cpp

This code implements a **Neural Network** specifically designed for **health score prediction**. Let's break down its purpose, functionality, and structure in detail:

---

### **Purpose of the Code**
The code is a **feedforward neural network** that predicts a health score based on multiple health-related features. It is designed to:
1. **Predict health scores**: Given a set of health-related features (e.g., blood pressure, cholesterol levels, etc.), the neural network predicts a health score that could represent overall health or risk of disease.
2. **Learn from data**: The network can be trained on a dataset of health features and corresponding health scores using **backpropagation** and **mini-batch gradient descent**.
3. **Normalize features**: It includes functionality to normalize input features (e.g., scaling values to a standard range) to improve training efficiency and accuracy.
4. **Support thread-safe inference**: The network can handle multiple prediction requests simultaneously in a thread-safe manner.

---

### **Main Functionality**
The code implements the following key components:

1. **Neural Network Architecture**:
   - The network consists of multiple **layers** (input, hidden, and output layers).
   - Each layer contains **neurons** that are connected to neurons in the next layer via **weights**.
   - The network supports **configurable activation functions** (e.g., Sigmoid, ReLU, Tanh, Linear) for each layer.

2. **Training**:
   - The network uses **backpropagation** to adjust weights during training.
   - It employs **mini-batch gradient descent** to update weights in small batches of data, which is more efficient than updating weights after every single data point.
   - **Early stopping** is supported to prevent overfitting (though not explicitly shown in the truncated code).

3. **Feature Normalization**:
   - Input features are normalized using either **Z-score normalization** (based on mean and standard deviation) or **min-max scaling** (scaling values to a range of [-1, 1]).
   - Normalization ensures that all features contribute equally to the prediction, regardless of their original scale.

4. **Thread-Safe Inference**:
   - A **mutex** (`inference_mutex`) is used to ensure that multiple threads can safely use the network for predictions without causing race conditions.

5. **Activation Functions**:
   - The network supports multiple activation functions, including:
     - **Sigmoid**: Smooth S-shaped curve, useful for binary classification.
     - **ReLU**: Rectified Linear Unit, commonly used in hidden layers.
     - **Tanh**: Hyperbolic tangent, similar to Sigmoid but outputs values in the range [-1, 1].
     - **Linear**: No transformation, outputs the input directly.

---

### **Algorithms Used**
1. **Feedforward Propagation**:
   - Input data is passed through the network layer by layer to compute the output (prediction).
   - Each neuron calculates a weighted sum of its inputs and applies an activation function.

2. **Backpropagation**:
   - During training, the network calculates the error between the predicted output and the actual target.
   - This error is propagated backward through the network to adjust the weights using the **gradient descent** algorithm.

3. **Mini-Batch Gradient Descent**:
   - Instead of updating weights after every single data point (stochastic gradient descent) or after the entire dataset (batch gradient descent), the network updates weights in small batches.
   - This balances efficiency and stability during training.

4. **Weight Initialization**:
   - Weights are initialized using **Xavier/Glorot initialization**, which scales the weights based on the number of input and output neurons. This helps prevent vanishing or exploding gradients.

5. **Feature Normalization**:
   - Input features are normalized using either Z-score normalization or min-max scaling to ensure consistent scaling across features.

---

### **Overall Structure**
The code is organized into several key components:

1. **Class Definition**:
   - The `HealthScorePredictor` class encapsulates the entire neural network.
   - It includes nested structures for `Feature`, `Neuron`, `Layer`, and `Connection`.

2. **Activation Functions**:
   - The `activate` and `activate_derivative` methods implement the activation functions and their derivatives, which are essential for forward and backward propagation.

3. **Feature Normalization**:
   - The `Feature` struct includes methods for normalizing and denormalizing input values.

4. **Neural Network Layers**:
   - The `Layer` struct represents a layer in the network, containing neurons and their connections.
   - The `Neuron` struct represents a single neuron, including its output value, gradient, and connections to the next layer.

5. **Thread Safety**:
   - A `std::mutex` (`inference_mutex`) ensures that the network can handle multiple prediction requests simultaneously without conflicts.

---

### **How the Parts Work Together**
1. **Input Data**:
   - Health-related features are passed to the network as input values.
   - These features are normalized using the `Feature` struct's `normalize` method.

2. **Forward Propagation**:
   - The `feed_forward` method computes the output of the network by passing the input values through each layer.
   - Each neuron calculates a weighted sum of its inputs and applies an activation function.

3. **Training**:
   - During training, the network uses backpropagation to adjust weights based on the error between predicted and actual health scores.
   - The `activate_derivative` method is used to compute gradients for weight updates.

4. **Prediction**:
   - Once trained, the network can predict health scores for new input data.
   - The `inference_mutex` ensures that predictions are thread-safe.

5. **Output**:
   - The final output is a predicted health score, which can be denormalized if necessary using the `Feature` struct's `denormalize` method.

---

### **Problem Being Solved**
The code addresses the problem of **predicting health scores** based on multiple health-related features. This is a common task in healthcare and wellness applications, where:
- Inputs might include metrics like blood pressure, cholesterol levels, BMI, etc.
- The output is a single score representing overall health or risk of disease.

The neural network approach is well-suited for this task because:
- It can capture complex, non-linear relationships between features.
- It can handle large datasets and high-dimensional input spaces.
- It can be trained to improve accuracy over time.

---

### **Summary**
This code is a **custom implementation of a neural network** for health score prediction. It includes:
- A flexible architecture with configurable layers and activation functions.
- Support for feature normalization and thread-safe inference.
- Training using backpropagation and mini-batch gradient descent.

The network is designed to be robust, efficient, and adaptable to different health prediction tasks.