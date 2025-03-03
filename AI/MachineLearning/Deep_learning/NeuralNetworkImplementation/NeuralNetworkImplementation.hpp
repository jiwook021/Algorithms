/**
 * @file NeuralNetworkImplementation.hpp
 * @brief From-scratch linear algebra (Vector, Matrix) and gradient descent.
 * @details Implements row-major Matrix, Vector with dot/transpose/multiply,
 *          used to demonstrate linear regression via gradient descent.
 */
#pragma once

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <iostream>

namespace Ml {

/**
 * @brief Dense column vector.
 */
class Vector {
public:
    explicit Vector(size_t size);
    Vector(const std::vector<double>& data);
    size_t Size() const;
    double& operator[](size_t i);
    const double& operator[](size_t i) const;
    Vector operator+(const Vector& other) const;
    Vector operator*(double scalar) const;
    double Dot(const Vector& other) const;

private:
    std::vector<double> Data_;
};

/**
 * @brief Row-major dense matrix.
 */
class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;
    size_t GetRows() const;
    size_t GetCols() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix Transpose() const;
    Vector operator*(const Vector& vec) const;

private:
    size_t Rows_, Cols_;
    std::vector<double> Data_;
};

/**
 * @brief Solve linear regression via gradient descent: min ||Xw - y||^2.
 * @return Learned weight vector.
 */
Vector LinearRegressionGD(const Matrix& x, const Vector& y,
                           double learningRate = 0.01, size_t iterations = 1000);

} // namespace Ml
