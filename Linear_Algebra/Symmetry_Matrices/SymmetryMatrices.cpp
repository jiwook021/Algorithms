/**
 * @file SymmetryMatrices.cpp
 * @brief Implementation of SymmetryMatrix class.
 * @details See SymmetryMatrices.hpp for the public API.
 */

#include "SymmetryMatrices.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

// ----------------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------------

SymmetryMatrix::SymmetryMatrix(
    const std::vector<std::vector<double>>& data)
    : matrix_(data), size_(data.size()) {
    if (size_ == 0)
        throw std::invalid_argument("Matrix size must be greater than 0.");
    for (const auto& row : matrix_)
        if (row.size() != size_)
            throw std::invalid_argument("Matrix must be square.");
}

SymmetryMatrix::SymmetryMatrix(std::size_t n, unsigned int seed)
    : size_(n) {
    if (n == 0)
        throw std::invalid_argument("Matrix size must be greater than 0.");

    matrix_.resize(n, std::vector<double>(n, 0.0));

    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            double val = dist(gen);
            matrix_[i][j] = val;
            if (i != j) matrix_[j][i] = val;
        }
    }
}

// ----------------------------------------------------------------------------
// Property queries
// ----------------------------------------------------------------------------

bool SymmetryMatrix::IsSymmetric() const {
    for (std::size_t i = 0; i < size_; ++i)
        for (std::size_t j = i + 1; j < size_; ++j)
            if (std::abs(matrix_[i][j] - matrix_[j][i]) > EPSILON)
                return false;
    return true;
}

double SymmetryMatrix::Trace() const {
    double sum = 0.0;
    for (std::size_t i = 0; i < size_; ++i) sum += matrix_[i][i];
    return sum;
}

double SymmetryMatrix::Determinant() const {
    if (size_ == 1) return matrix_[0][0];
    if (size_ == 2)
        return matrix_[0][0] * matrix_[1][1] -
               matrix_[0][1] * matrix_[1][0];
    if (size_ == 3) {
        return matrix_[0][0] * (matrix_[1][1] * matrix_[2][2] -
                                matrix_[1][2] * matrix_[2][1]) -
               matrix_[0][1] * (matrix_[1][0] * matrix_[2][2] -
                                matrix_[1][2] * matrix_[2][0]) +
               matrix_[0][2] * (matrix_[1][0] * matrix_[2][1] -
                                matrix_[1][1] * matrix_[2][0]);
    }
    throw std::logic_error(
        "Determinant for 4x4 or larger matrices is not implemented.");
}

double SymmetryMatrix::FrobeniusNorm() const {
    double sum = 0.0;
    for (const auto& row : matrix_)
        for (double v : row) sum += v * v;
    return std::sqrt(sum);
}

double SymmetryMatrix::ApproximateLargestEigenvalue(
    std::size_t iterations) const {
    std::vector<double> v(size_, 1.0);
    double norm = std::sqrt(std::accumulate(
        v.begin(), v.end(), 0.0,
        [](double s, double x) { return s + x * x; }));
    for (auto& val : v) val /= norm;

    for (std::size_t iter = 0; iter < iterations; ++iter) {
        std::vector<double> newV(size_, 0.0);
        for (std::size_t i = 0; i < size_; ++i)
            for (std::size_t j = 0; j < size_; ++j)
                newV[i] += matrix_[i][j] * v[j];

        norm = std::sqrt(std::accumulate(
            newV.begin(), newV.end(), 0.0,
            [](double s, double x) { return s + x * x; }));
        for (auto& val : newV) val /= norm;
        v = newV;
    }

    // Rayleigh quotient.
    double num = 0.0;
    for (std::size_t i = 0; i < size_; ++i)
        for (std::size_t j = 0; j < size_; ++j)
            num += v[i] * matrix_[i][j] * v[j];
    double den = std::accumulate(
        v.begin(), v.end(), 0.0,
        [](double s, double x) { return s + x * x; });
    return num / den;
}

bool SymmetryMatrix::IsPositiveDefinite() const {
    for (std::size_t i = 0; i < size_; ++i)
        if (matrix_[i][i] <= 0) return false;

    if (size_ == 1) return matrix_[0][0] > 0;
    if (size_ == 2) {
        double det = matrix_[0][0] * matrix_[1][1] -
                     matrix_[0][1] * matrix_[1][0];
        return matrix_[0][0] > 0 && det > 0;
    }
    if (size_ == 3) {
        double d1 = matrix_[0][0];
        double d2 = matrix_[0][0] * matrix_[1][1] -
                     matrix_[0][1] * matrix_[1][0];
        double d3 = Determinant();
        return d1 > 0 && d2 > 0 && d3 > 0;
    }
    throw std::logic_error(
        "Positive definiteness check for n > 3 is not implemented.");
}

// ----------------------------------------------------------------------------
// Arithmetic
// ----------------------------------------------------------------------------

SymmetryMatrix SymmetryMatrix::Add(const SymmetryMatrix& other) const {
    if (size_ != other.size_)
        throw std::invalid_argument("Matrix sizes do not match.");
    std::vector<std::vector<double>> result(
        size_, std::vector<double>(size_));
    for (std::size_t i = 0; i < size_; ++i)
        for (std::size_t j = 0; j < size_; ++j)
            result[i][j] = matrix_[i][j] + other.matrix_[i][j];
    return SymmetryMatrix(result);
}

std::vector<std::vector<double>> SymmetryMatrix::Multiply(
    const SymmetryMatrix& other) const {
    if (size_ != other.size_)
        throw std::invalid_argument("Matrix sizes do not match.");
    std::vector<std::vector<double>> result(
        size_, std::vector<double>(size_, 0.0));
    for (std::size_t i = 0; i < size_; ++i)
        for (std::size_t j = 0; j < size_; ++j)
            for (std::size_t k = 0; k < size_; ++k)
                result[i][j] += matrix_[i][k] * other.matrix_[k][j];
    return result;
}

bool SymmetryMatrix::IsProductSymmetric(
    const SymmetryMatrix& other) const {
    auto product = Multiply(other);
    for (std::size_t i = 0; i < size_; ++i)
        for (std::size_t j = i + 1; j < size_; ++j)
            if (std::abs(product[i][j] - product[j][i]) > EPSILON)
                return false;
    return true;
}

bool SymmetryMatrix::IsCommutative(const SymmetryMatrix& other) const {
    auto ab = Multiply(other);
    auto ba = other.Multiply(*this);
    for (std::size_t i = 0; i < size_; ++i)
        for (std::size_t j = 0; j < size_; ++j)
            if (std::abs(ab[i][j] - ba[i][j]) > EPSILON)
                return false;
    return true;
}

// ----------------------------------------------------------------------------
// Accessors
// ----------------------------------------------------------------------------

double SymmetryMatrix::GetValue(std::size_t i, std::size_t j) const {
    if (i >= size_ || j >= size_)
        throw std::out_of_range("Index out of range.");
    return matrix_[i][j];
}

void SymmetryMatrix::SetValue(std::size_t i, std::size_t j, double value) {
    if (i >= size_ || j >= size_)
        throw std::out_of_range("Index out of range.");
    matrix_[i][j] = value;
}

std::size_t SymmetryMatrix::GetSize() const { return size_; }

void SymmetryMatrix::Print() const {
    for (const auto& row : matrix_) {
        for (double v : row)
            std::cout << std::setw(10) << std::fixed << std::setprecision(2)
                      << v << " ";
        std::cout << "\n";
    }
}

// ----------------------------------------------------------------------------
// Free function
// ----------------------------------------------------------------------------

SymmetryMatrix CreateIdentityMatrix(std::size_t n) {
    std::vector<std::vector<double>> data(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) data[i][i] = 1.0;
    return SymmetryMatrix(data);
}
