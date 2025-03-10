#include <vector>
#include <stdexcept>
#include <iostream>

// Vector class definition
class Vector {
private:
    std::vector<double> data;

public:
    // Constructor with size, initialized to zeros
    Vector(size_t size) : data(size, 0.0) {}
    
    // Constructor from an existing std::vector
    Vector(const std::vector<double>& vec) : data(vec) {}
    
    // Get the size of the vector
    size_t size() const { return data.size(); }
    
    // Access and modify elements
    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }
    
    // Vector addition
    Vector operator+(const Vector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vectors must have the same size for addition");
        }
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] + other[i];
        }
        return result;
    }
    
    // Scalar multiplication
    Vector operator*(double scalar) const {
        Vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = data[i] * scalar;
        }
        return result;
    }
    
    // Dot product
    double dot(const Vector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Vectors must have the same size for dot product");
        }
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            sum += data[i] * other[i];
        }
        return sum;
    }
};

// Matrix class definition
class Matrix {
private:
    size_t rows;
    size_t cols;
    std::vector<double> data; // Stored in row-major order

public:
    // Constructor with rows and columns, initialized to zeros
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0) {}
    
    // Access and modify elements using (i, j) notation
    double& operator()(size_t i, size_t j) {
        return data[i * cols + j];
    }
    const double& operator()(size_t i, size_t j) const {
        return data[i * cols + j];
    }
    
    // Matrix addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices must have the same dimensions for addition");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return result;
    }
    
    // Matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Number of columns of first matrix must equal number of rows of second for multiplication");
        }
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }
    
    // Matrix transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }
    
    // Matrix-vector multiplication
    Vector operator*(const Vector& vec) const {
        if (cols != vec.size()) {
            throw std::invalid_argument("Matrix columns must equal vector size for multiplication");
        }
        Vector result(rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i] += (*this)(i, j) * vec[j];
            }
        }
        return result;
    }
};

// Main function implementing linear regression with gradient descent
int main() {
    // Define feature matrix X (3 samples, 2 features)
    Matrix X(3, 2);
    X(0, 0) = 1; X(0, 1) = 2;
    X(1, 0) = 3; X(1, 1) = 4;
    X(2, 0) = 5; X(2, 1) = 6;
    
    // Define target vector y
    Vector y(3);
    y[0] = 3.5;  // 1*0.5 + 2*1.5
    y[1] = 7.5;  // 3*0.5 + 4*1.5
    y[2] = 11.5; // 5*0.5 + 6*1.5
    
    // Initialize weights to zero
    Vector w(2);
    w[0] = 0.0; w[1] = 0.0;
    
    // Gradient descent parameters
    double learning_rate = 0.01;
    size_t num_iterations = 1000;
    size_t n = 3; // Number of samples
    
    // Gradient descent loop
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        // Prediction: X * w
        Vector prediction = X * w;
        
        // Error: prediction - y
        Vector error = prediction + (y * (-1.0));
        
        // Gradient: (2/n) * X^T * error
        Matrix X_transpose = X.transpose();
        Vector gradient = X_transpose * error;
        gradient = gradient * (2.0 / static_cast<double>(n));
        
        // Update weights: w = w - learning_rate * gradient
        w = w + (gradient * (-learning_rate));
    }
    
    // Output the learned weights
    std::cout << "Learned weights: " << w[0] << ", " << w[1] << std::endl;
    
    return 0;
}
//https://claude.ai/chat/26b39d97-541b-465e-acc4-1c40b2054852