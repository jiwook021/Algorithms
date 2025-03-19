# Code Overview: main.cpp

The purpose of this C++ code is to implement a simple linear regression model using gradient descent. Linear regression is a fundamental algorithm in machine learning used for predicting a continuous target variable based on one or more input features. The code is structured to define custom `Vector` and `Matrix` classes to handle mathematical operations, and it uses these classes to perform linear regression on a small dataset.

### Main Functionality

The code performs linear regression using gradient descent to find the optimal weights that minimize the error between the predicted and actual target values. The main components of the code include:

1. **Custom Vector and Matrix Classes**: These classes provide the necessary operations for vector and matrix arithmetic, such as addition, scalar multiplication, dot product, and matrix multiplication. They are essential for implementing the linear algebra operations required in the gradient descent algorithm.

2. **Gradient Descent Algorithm**: This is an iterative optimization algorithm used to minimize the cost function (mean squared error in this case) by updating the weights in the direction of the steepest descent, which is determined by the gradient of the cost function.

### Problem Being Solved

The problem being solved is a linear regression task where the goal is to find the best-fitting line (or hyperplane in higher dimensions) that predicts the target variable `y` based on the input features `X`. The dataset consists of 3 samples with 2 features each, and the target values are provided.

### Approach Taken

1. **Data Representation**: The input features are stored in a `Matrix` object `X`, and the target values are stored in a `Vector` object `y`. The weights (or coefficients) of the linear model are also stored in a `Vector` object `w`.

2. **Initialization**: The weights are initialized to zero, and the learning rate and number of iterations for gradient descent are set.

3. **Gradient Descent Loop**:
   - **Prediction**: Calculate the predicted values by multiplying the feature matrix `X` with the weight vector `w`.
   - **Error Calculation**: Compute the error by subtracting the actual target values `y` from the predicted values.
   - **Gradient Calculation**: Compute the gradient of the cost function with respect to the weights. This involves transposing the feature matrix `X`, multiplying it by the error vector, and scaling by `2/n` (where `n` is the number of samples).
   - **Weight Update**: Update the weights by subtracting the product of the learning rate and the gradient from the current weights.

4. **Output**: After the specified number of iterations, the learned weights are printed, representing the coefficients of the linear regression model.

### Overall Structure

- **Vector Class**: Handles operations on vectors, including element access, addition, scalar multiplication, and dot product.
- **Matrix Class**: Handles operations on matrices, including element access, addition, multiplication, transpose, and matrix-vector multiplication.
- **Main Function**: Sets up the data, initializes parameters, runs the gradient descent algorithm, and outputs the learned weights.

The code effectively demonstrates the use of basic linear algebra operations to implement a simple machine learning algorithm, showcasing how custom data structures can be used to perform mathematical computations in C++.