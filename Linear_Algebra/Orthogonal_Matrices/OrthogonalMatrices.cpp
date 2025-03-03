/**
 * @file OrthogonalMatrices.cpp
 * @brief Implementation of Gram-Schmidt and orthogonal-matrix utilities.
 * @details See OrthogonalMatrices.hpp for the public API.
 */

#include "OrthogonalMatrices.hpp"

#include <cmath>
#include <stdexcept>

// ----------------------------------------------------------------------------
// Vector operations
// ----------------------------------------------------------------------------

double OrthogonalMatrices::DotProduct(const Vector& a, const Vector& b) {
    if (a.size() != b.size())
        throw std::invalid_argument(
            "Vectors must have the same dimension for dot product.");
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

double OrthogonalMatrices::Norm(const Vector& v) {
    return std::sqrt(DotProduct(v, v));
}

OrthogonalMatrices::Vector OrthogonalMatrices::ScalarMultiply(
    const Vector& v, double scalar) {
    Vector result(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) result[i] = v[i] * scalar;
    return result;
}

OrthogonalMatrices::Vector OrthogonalMatrices::SubtractVectors(
    const Vector& a, const Vector& b) {
    if (a.size() != b.size())
        throw std::invalid_argument(
            "Vectors must have the same dimension for subtraction.");
    Vector result(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
    return result;
}

// ----------------------------------------------------------------------------
// Gram-Schmidt
// ----------------------------------------------------------------------------

OrthogonalMatrices::Matrix OrthogonalMatrices::GramSchmidt(
    const Matrix& inputVectors) {
    Matrix basis;
    for (const auto& v : inputVectors) {
        if (v.empty())
            throw std::invalid_argument("Input vector is empty.");

        Vector orthVec = v;
        for (const auto& basisVec : basis) {
            double coeff = DotProduct(orthVec, basisVec);
            auto proj = ScalarMultiply(basisVec, coeff);
            orthVec = SubtractVectors(orthVec, proj);
        }

        double norm = Norm(orthVec);
        if (norm < ORTHOGONAL_EPSILON) continue;  // linearly dependent

        basis.push_back(ScalarMultiply(orthVec, 1.0 / norm));
    }
    return basis;
}

// ----------------------------------------------------------------------------
// Matrix operations
// ----------------------------------------------------------------------------

OrthogonalMatrices::Matrix OrthogonalMatrices::MultiplyMatrices(
    const Matrix& a, const Matrix& b) {
    if (a.empty() || b.empty())
        throw std::invalid_argument("Empty matrix provided.");
    std::size_t m = a.size(), k = a[0].size(), n = b[0].size();
    if (k != b.size())
        throw std::invalid_argument(
            "Incompatible matrix dimensions for multiplication.");

    Matrix c(m, Vector(n, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t l = 0; l < k; ++l)
                c[i][j] += a[i][l] * b[l][j];
    return c;
}

OrthogonalMatrices::Matrix OrthogonalMatrices::TransposeMatrix(
    const Matrix& m) {
    if (m.empty())
        throw std::invalid_argument("Empty matrix provided.");
    std::size_t rows = m.size(), cols = m[0].size();
    Matrix t(cols, Vector(rows, 0.0));
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            t[j][i] = m[i][j];
    return t;
}

bool OrthogonalMatrices::IsOrthogonalMatrix(const Matrix& m) {
    if (m.empty())
        throw std::invalid_argument("Empty matrix provided.");
    std::size_t n = m.size();
    for (const auto& row : m)
        if (row.size() != n)
            throw std::invalid_argument("Matrix is not square.");

    auto mt = TransposeMatrix(m);
    auto product = MultiplyMatrices(m, mt);

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(product[i][j] - expected) > ORTHOGONAL_EPSILON)
                return false;
        }
    return true;
}
