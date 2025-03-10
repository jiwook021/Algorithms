#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    // Constructors
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    Matrix(const std::vector<std::vector<double>>& mat) : data(mat) {
        if (mat.empty()) {
            rows = 0;
            cols = 0;
        } else {
            rows = mat.size();
            cols = mat[0].size();
            // Ensure all rows have the same number of columns
            for (const auto& row : mat) {
                if (row.size() != cols) {
                    throw std::invalid_argument("All rows must have the same number of columns");
                }
            }
        }
    }

    // Access elements
    double& at(size_t i, size_t j) {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[i][j];
    }

    double at(size_t i, size_t j) const {
        if (i >= rows || j >= cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return data[i][j];
    }

    // Get dimensions
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // Matrix operations
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(j, i) = data[i][j];
            }
        }
        return result;
    }

    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = data[i][j] + other.at(i, j);
            }
        }
        return result;
    }

    // Matrix subtraction
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = data[i][j] - other.at(i, j);
            }
        }
        return result;
    }

    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }

        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * other.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
        return result;
    }

    // Scalar multiplication
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }

    // Hadamard product (element-wise multiplication)
    Matrix hadamard(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for Hadamard product");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = data[i][j] * other.at(i, j);
            }
        }
        return result;
    }

    // Matrix initialization methods
    static Matrix zeros(size_t rows, size_t cols) {
        return Matrix(rows, cols);
    }

    static Matrix ones(size_t rows, size_t cols) {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = 1.0;
            }
        }
        return result;
    }

    static Matrix identity(size_t n) {
        Matrix result(n, n);
        for (size_t i = 0; i < n; ++i) {
            result.at(i, i) = 1.0;
        }
        return result;
    }

    static Matrix random(size_t rows, size_t cols, double min = 0.0, double max = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(min, max);

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = dist(gen);
            }
        }
        return result;
    }

    // Mathematical functions applied element-wise
    Matrix apply(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = func(data[i][j]);
            }
        }
        return result;
    }

    // Common activation functions for neural networks
    Matrix sigmoid() const {
        auto sigmoid_func = [](double x) -> double {
            return 1.0 / (1.0 + std::exp(-x));
        };
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = sigmoid_func(data[i][j]);
            }
        }
        return result;
    }

    Matrix relu() const {
        auto relu_func = [](double x) -> double {
            return std::max(0.0, x);
        };
        
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(i, j) = relu_func(data[i][j]);
            }
        }
        return result;
    }

    // Print matrix
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
        for (size_t i = 0; i < matrix.rows; ++i) {
            os << "[";
            for (size_t j = 0; j < matrix.cols; ++j) {
                os << matrix.data[i][j];
                if (j < matrix.cols - 1) {
                    os << ", ";
                }
            }
            os << "]" << std::endl;
        }
        return os;
    }
};

// Vector class (special case of matrix with 1 column)
class Vector {
private:
    std::vector<double> data;
    size_t size;

public:
    // Constructor
    Vector(size_t size) : size(size) {
        data.resize(size, 0.0);
    }

    Vector(const std::vector<double>& vec) : data(vec), size(vec.size()) {}

    // Convert to/from Matrix
    Matrix toMatrix() const {
        std::vector<std::vector<double>> mat(size, std::vector<double>(1));
        for (size_t i = 0; i < size; ++i) {
            mat[i][0] = data[i];
        }
        return Matrix(mat);
    }

    static Vector fromMatrix(const Matrix& mat) {
        if (mat.getCols() != 1) {
            throw std::invalid_argument("Matrix must have exactly 1 column to convert to Vector");
        }
        
        Vector vec(mat.getRows());
        for (size_t i = 0; i < mat.getRows(); ++i) {
            vec.at(i) = mat.at(i, 0);
        }
        return vec;
    }

    // Access elements
    double& at(size_t i) {
        if (i >= size) {
            throw std::out_of_range("Vector index out of range");
        }
        return data[i];
    }

    double at(size_t i) const {
        if (i >= size) {
            throw std::out_of_range("Vector index out of range");
        }
        return data[i];
    }

    // Get size
    size_t getSize() const { return size; }

    // Vector operations
    double dot(const Vector& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions don't match for dot product");
        }

        double result = 0.0;
        for (size_t i = 0; i < size; ++i) {
            result += data[i] * other.data[i];
        }
        return result;
    }

    double norm() const {
        double sum = 0.0;
        for (const auto& val : data) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    Vector normalize() const {
        double n = norm();
        if (n == 0) {
            throw std::runtime_error("Cannot normalize a zero vector");
        }

        Vector result(size);
        for (size_t i = 0; i < size; ++i) {
            result.at(i) = data[i] / n;
        }
        return result;
    }

    // Vector arithmetic
    Vector operator+(const Vector& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions don't match for addition");
        }

        Vector result(size);
        for (size_t i = 0; i < size; ++i) {
            result.at(i) = data[i] + other.data[i];
        }
        return result;
    }

    Vector operator-(const Vector& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions don't match for subtraction");
        }

        Vector result(size);
        for (size_t i = 0; i < size; ++i) {
            result.at(i) = data[i] - other.data[i];
        }
        return result;
    }

    Vector operator*(double scalar) const {
        Vector result(size);
        for (size_t i = 0; i < size; ++i) {
            result.at(i) = data[i] * scalar;
        }
        return result;
    }

    // Print vector
    friend std::ostream& operator<<(std::ostream& os, const Vector& vector) {
        os << "[";
        for (size_t i = 0; i < vector.size; ++i) {
            os << vector.data[i];
            if (i < vector.size - 1) {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
};

// Machine Learning Algorithms

// Linear Regression
class LinearRegression {
private:
    Vector weights;
    double bias;
    double learning_rate;
    int max_iterations;

public:
    LinearRegression(size_t n_features, double learning_rate = 0.01, int max_iterations = 1000)
        : weights(n_features), bias(0.0), learning_rate(learning_rate), max_iterations(max_iterations) {
        // Initialize weights with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        
        for (size_t i = 0; i < n_features; ++i) {
            weights.at(i) = dist(gen);
        }
    }

    // Predict function (y = w*x + b)
    double predict(const Vector& x) const {
        return weights.dot(x) + bias;
    }

    // Train the model using gradient descent
    void fit(const std::vector<Vector>& X, const std::vector<double>& y, bool verbose = false) {
        if (X.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid training data");
        }

        size_t n_samples = X.size();
        
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            double total_error = 0.0;
            
            // Gradient for weights and bias
            Vector dw(weights.getSize());
            double db = 0.0;
            
            // Compute gradients
            for (size_t i = 0; i < n_samples; ++i) {
                double prediction = predict(X[i]);
                double error = prediction - y[i];
                total_error += error * error;
                
                // Update gradients
                for (size_t j = 0; j < weights.getSize(); ++j) {
                    dw.at(j) += error * X[i].at(j);
                }
                db += error;
            }
            
            // Update weights and bias
            for (size_t j = 0; j < weights.getSize(); ++j) {
                weights.at(j) -= learning_rate * dw.at(j) / n_samples;
            }
            bias -= learning_rate * db / n_samples;
            
            // Calculate mean squared error
            double mse = total_error / n_samples;
            
            if (verbose && (iteration % 100 == 0)) {
                std::cout << "Iteration " << iteration << ": MSE = " << mse << std::endl;
            }
        }
    }

    // Get the trained weights
    Vector getWeights() const {
        return weights;
    }

    // Get the trained bias
    double getBias() const {
        return bias;
    }
};

// Logistic Regression for binary classification
class LogisticRegression {
private:
    Vector weights;
    double bias;
    double learning_rate;
    int max_iterations;

    // Sigmoid function
    double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

public:
    LogisticRegression(size_t n_features, double learning_rate = 0.01, int max_iterations = 1000)
        : weights(n_features), bias(0.0), learning_rate(learning_rate), max_iterations(max_iterations) {
        // Initialize weights with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        
        for (size_t i = 0; i < n_features; ++i) {
            weights.at(i) = dist(gen);
        }
    }

    // Predict probability (p(y=1|x) = sigmoid(w*x + b))
    double predict_proba(const Vector& x) const {
        return sigmoid(weights.dot(x) + bias);
    }

    // Predict class (0 or 1)
    int predict(const Vector& x) const {
        return predict_proba(x) >= 0.5 ? 1 : 0;
    }

    // Train the model using gradient descent
    void fit(const std::vector<Vector>& X, const std::vector<int>& y, bool verbose = false) {
        if (X.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid training data");
        }

        size_t n_samples = X.size();
        
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            double total_loss = 0.0;
            
            // Gradient for weights and bias
            Vector dw(weights.getSize());
            double db = 0.0;
            
            // Compute gradients
            for (size_t i = 0; i < n_samples; ++i) {
                double prob = predict_proba(X[i]);
                double error = prob - y[i];
                
                // Binary cross-entropy loss
                if (y[i] == 1) {
                    total_loss -= std::log(prob);
                } else {
                    total_loss -= std::log(1 - prob);
                }
                
                // Update gradients
                for (size_t j = 0; j < weights.getSize(); ++j) {
                    dw.at(j) += error * X[i].at(j);
                }
                db += error;
            }
            
            // Update weights and bias
            for (size_t j = 0; j < weights.getSize(); ++j) {
                weights.at(j) -= learning_rate * dw.at(j) / n_samples;
            }
            bias -= learning_rate * db / n_samples;
            
            // Calculate mean loss
            double mean_loss = total_loss / n_samples;
            
            if (verbose && (iteration % 100 == 0)) {
                std::cout << "Iteration " << iteration << ": Loss = " << mean_loss << std::endl;
            }
        }
    }

    // Get the trained weights
    Vector getWeights() const {
        return weights;
    }

    // Get the trained bias
    double getBias() const {
        return bias;
    }
};

// Simple Neural Network (Multi-Layer Perceptron)
class NeuralNetwork {
private:
    std::vector<Matrix> weights;
    std::vector<Vector> biases;
    double learning_rate;
    int max_iterations;

    // Apply sigmoid to a matrix
    Matrix sigmoid(const Matrix& z) const {
        return z.sigmoid();
    }

    // Derivative of sigmoid
    Matrix sigmoid_derivative(const Matrix& a) const {
        Matrix ones = Matrix::ones(a.getRows(), a.getCols());
        return a.hadamard(ones - a);
    }

public:
    NeuralNetwork(const std::vector<size_t>& layer_sizes, double learning_rate = 0.01, int max_iterations = 1000)
        : learning_rate(learning_rate), max_iterations(max_iterations) {
        
        if (layer_sizes.size() < 2) {
            throw std::invalid_argument("Neural network must have at least 2 layers");
        }

        // Initialize weights and biases
        for (size_t i = 1; i < layer_sizes.size(); ++i) {
            // Initialize weights with small random values
            Matrix w = Matrix::random(layer_sizes[i], layer_sizes[i-1], -0.1, 0.1);
            weights.push_back(w);
            
            // Initialize biases with zeros
            Vector b(layer_sizes[i]);
            biases.push_back(b);
        }
    }

    // Forward pass
    Matrix forward(const Matrix& X) const {
        Matrix activation = X;
        
        for (size_t i = 0; i < weights.size(); ++i) {
            Matrix z = weights[i] * activation.transpose();
            
            // Add bias to each column
            for (size_t j = 0; j < z.getRows(); ++j) {
                for (size_t k = 0; k < z.getCols(); ++k) {
                    z.at(j, k) += biases[i].at(j);
                }
            }
            
            activation = sigmoid(z).transpose();
        }
        
        return activation;
    }

    // Train the model using backpropagation
    void fit(const Matrix& X, const Matrix& y, bool verbose = false) {
        size_t n_samples = X.getRows();
        
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            // Forward pass
            std::vector<Matrix> activations;
            std::vector<Matrix> z_values;
            
            activations.push_back(X);
            
            Matrix activation = X;
            for (size_t i = 0; i < weights.size(); ++i) {
                Matrix z = weights[i] * activation.transpose();
                
                // Add bias to each column
                for (size_t j = 0; j < z.getRows(); ++j) {
                    for (size_t k = 0; k < z.getCols(); ++k) {
                        z.at(j, k) += biases[i].at(j);
                    }
                }
                
                z_values.push_back(z);
                activation = sigmoid(z).transpose();
                activations.push_back(activation);
            }
            
            // Compute loss
            double total_loss = 0.0;
            Matrix output = activations.back();
            for (size_t i = 0; i < n_samples; ++i) {
                for (size_t j = 0; j < y.getCols(); ++j) {
                    double pred = output.at(i, j);
                    double target = y.at(i, j);
                    total_loss += -target * std::log(pred) - (1 - target) * std::log(1 - pred);
                }
            }
            
            // Backpropagation
            std::vector<Matrix> deltas;
            
            // Output layer error
            Matrix error = output - y;
            deltas.push_back(error);
            
            // Hidden layers error
            for (int i = weights.size() - 2; i >= 0; --i) {
                Matrix delta = (weights[i+1].transpose() * deltas.back().transpose()).transpose().hadamard(
                    sigmoid_derivative(activations[i+1]));
                deltas.push_back(delta);
            }
            
            // Reverse deltas to match weights order
            std::reverse(deltas.begin(), deltas.end());
            
            // Update weights and biases
            for (size_t i = 0; i < weights.size(); ++i) {
                Matrix dw = deltas[i].transpose() * activations[i];
                
                // Update weights
                for (size_t j = 0; j < weights[i].getRows(); ++j) {
                    for (size_t k = 0; k < weights[i].getCols(); ++k) {
                        weights[i].at(j, k) -= learning_rate * dw.at(j, k) / n_samples;
                    }
                }
                
                // Update biases
                for (size_t j = 0; j < biases[i].getSize(); ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < deltas[i].getRows(); ++k) {
                        sum += deltas[i].at(k, j);
                    }
                    biases[i].at(j) -= learning_rate * sum / n_samples;
                }
            }
            
            if (verbose && (iteration % 100 == 0)) {
                double mean_loss = total_loss / n_samples;
                std::cout << "Iteration " << iteration << ": Loss = " << mean_loss << std::endl;
            }
        }
    }

    // Predict
    Matrix predict(const Matrix& X) const {
        return forward(X);
    }
};

// Example usage
int main() {
    // Matrix operations example
    std::cout << "Matrix Operations Example:" << std::endl;
    Matrix A({{1, 2}, {3, 4}});
    Matrix B({{5, 6}, {7, 8}});
    
    std::cout << "Matrix A:" << std::endl << A;
    std::cout << "Matrix B:" << std::endl << B;
    std::cout << "A + B:" << std::endl << (A + B);
    std::cout << "A * B:" << std::endl << (A * B);
    std::cout << "A transposed:" << std::endl << A.transpose();
    
    // Linear Regression example
    std::cout << "\nLinear Regression Example:" << std::endl;
    
    // Generate synthetic data: y = 2*x1 + 3*x2 + noise
    std::vector<Vector> X_train;
    std::vector<double> y_train;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 0.5);  // Noise with mean 0 and std 0.5
    
    for (int i = 0; i < 100; ++i) {
        Vector sample(2);
        sample.at(0) = i / 50.0;
        sample.at(1) = i / 25.0;
        
        X_train.push_back(sample);
        y_train.push_back(2 * sample.at(0) + 3 * sample.at(1) + dist(gen));
    }
    
    // Train linear regression model
    LinearRegression lr(2, 0.1, 1000);
    lr.fit(X_train, y_train, true);
    
    std::cout << "True coefficients: w1=2, w2=3, b=0" << std::endl;
    std::cout << "Learned coefficients: w1=" << lr.getWeights().at(0)
              << ", w2=" << lr.getWeights().at(1)
              << ", b=" << lr.getBias() << std::endl;
              
    // Test prediction
    Vector test_sample(2);
    test_sample.at(0) = 1.5;
    test_sample.at(1) = 2.0;
    
    std::cout << "Prediction for [1.5, 2.0]: " << lr.predict(test_sample) << std::endl;
    std::cout << "Expected value: " << (2 * 1.5 + 3 * 2.0) << std::endl;
    
    return 0;
}