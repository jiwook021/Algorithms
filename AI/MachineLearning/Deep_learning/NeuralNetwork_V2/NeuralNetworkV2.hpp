/**
 * @file NeuralNetworkV2.hpp
 * @brief Matrix-based neural network with linear and logistic regression
 * @details Implements Matrix/Vector math, LinearRegression, LogisticRegression,
 *          and a multi-layer perceptron using matrix operations and backpropagation.
 */
#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <functional>

namespace Ml {

/**
 * @brief Dense matrix with element-wise and algebraic operations.
 */
class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    Matrix(const std::vector<std::vector<double>>& mat);

    double& At(size_t i, size_t j);
    double  At(size_t i, size_t j) const;
    size_t GetRows() const { return Rows_; }
    size_t GetCols() const { return Cols_; }

    Matrix Transpose() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix Hadamard(const Matrix& other) const;
    Matrix Apply(std::function<double(double)> fn) const;
    Matrix Sigmoid() const;
    Matrix Relu() const;

    static Matrix Zeros(size_t rows, size_t cols);
    static Matrix Ones(size_t rows, size_t cols);
    static Matrix Identity(size_t n);
    static Matrix Random(size_t rows, size_t cols, double min = 0.0, double max = 1.0);

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

private:
    std::vector<std::vector<double>> Data_;
    size_t Rows_;
    size_t Cols_;
};

/**
 * @brief Column vector helper (wraps std::vector<double>).
 */
class Vector {
public:
    explicit Vector(size_t size);
    Vector(const std::vector<double>& vec);

    double& At(size_t i);
    double  At(size_t i) const;
    size_t GetSize() const { return Size_; }

    Matrix ToMatrix() const;
    static Vector FromMatrix(const Matrix& mat);

    double Dot(const Vector& other) const;
    double Norm() const;
    Vector Normalize() const;

    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector operator*(double scalar) const;

    friend std::ostream& operator<<(std::ostream& os, const Vector& v);

private:
    std::vector<double> Data_;
    size_t Size_;
};

/**
 * @brief Linear regression via gradient descent.
 */
class LinearRegression {
public:
    LinearRegression(size_t nFeatures, double learningRate = 0.01, int maxIter = 1000);
    double Predict(const Vector& x) const;
    void Fit(const std::vector<Vector>& x, const std::vector<double>& y, bool verbose = false);
    Vector GetWeights() const;
    double GetBias() const;

private:
    Vector Weights_;
    double Bias_;
    double LearningRate_;
    int MaxIterations_;
};

/**
 * @brief Logistic regression for binary classification.
 */
class LogisticRegression {
public:
    LogisticRegression(size_t nFeatures, double learningRate = 0.01, int maxIter = 1000);
    double PredictProba(const Vector& x) const;
    int Predict(const Vector& x) const;
    void Fit(const std::vector<Vector>& x, const std::vector<int>& y, bool verbose = false);
    Vector GetWeights() const;
    double GetBias() const;

private:
    double SigmoidFn(double x) const;
    Vector Weights_;
    double Bias_;
    double LearningRate_;
    int MaxIterations_;
};

/**
 * @brief Multi-layer perceptron with sigmoid activations and matrix math.
 */
class NeuralNetworkV2 {
public:
    NeuralNetworkV2(const std::vector<size_t>& layerSizes,
                    double learningRate = 0.01, int maxIter = 1000);

    Matrix Forward(const Matrix& x) const;
    void Fit(const Matrix& x, const Matrix& y, bool verbose = false);
    Matrix Predict(const Matrix& x) const;

private:
    Matrix SigmoidMat(const Matrix& z) const;
    Matrix SigmoidDerivative(const Matrix& a) const;

    std::vector<Matrix> Weights_;
    std::vector<Vector> Biases_;
    double LearningRate_;
    int MaxIterations_;
};

} // namespace Ml
