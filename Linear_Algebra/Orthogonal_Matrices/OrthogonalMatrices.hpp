/**
 * @file OrthogonalMatrices.hpp
 * @brief Gram-Schmidt orthonormalization and orthogonal-matrix verification.
 * @details Provides functions to produce an orthonormal basis from a set of
 *          vectors using the classical Gram-Schmidt process and to check
 *          whether a given square matrix is orthogonal (Q Q^T = I).
 */

#ifndef ORTHOGONAL_MATRICES_HPP
#define ORTHOGONAL_MATRICES_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

/// Tolerance for floating-point comparisons.
constexpr double ORTHOGONAL_EPSILON = 1e-9;

/**
 * @brief Utilities for orthogonal matrices and orthonormalization.
 */
class OrthogonalMatrices {
public:
    using Vector = std::vector<double>;
    using Matrix = std::vector<std::vector<double>>;

    /**
     * @brief Dot product of two vectors.
     * @throws std::invalid_argument if dimensions differ.
     * @details Time O(n), Space O(1).
     */
    static double DotProduct(const Vector& a, const Vector& b);

    /**
     * @brief Euclidean (L2) norm.
     * @details Time O(n), Space O(1).
     */
    static double Norm(const Vector& v);

    /**
     * @brief Multiply a vector by a scalar.
     * @details Time O(n), Space O(n).
     */
    static Vector ScalarMultiply(const Vector& v, double scalar);

    /**
     * @brief Element-wise subtraction a - b.
     * @throws std::invalid_argument if dimensions differ.
     * @details Time O(n), Space O(n).
     */
    static Vector SubtractVectors(const Vector& a, const Vector& b);

    /**
     * @brief Compute an orthonormal basis via Gram-Schmidt.
     * @param inputVectors  Set of input vectors.
     * @return Orthonormal basis vectors (linearly dependent inputs skipped).
     * @throws std::invalid_argument if any vector is empty.
     *
     * @details Time O(k^2 * n), Space O(k * n),
     *          where k = number of vectors, n = dimension.
     */
    static Matrix GramSchmidt(const Matrix& inputVectors);

    /**
     * @brief Matrix multiplication A * B.
     * @throws std::invalid_argument if dimensions are incompatible.
     * @details Time O(m*n*p), Space O(m*p).
     */
    static Matrix MultiplyMatrices(const Matrix& a, const Matrix& b);

    /**
     * @brief Transpose a matrix.
     * @throws std::invalid_argument if the matrix is empty.
     * @details Time O(m*n), Space O(n*m).
     */
    static Matrix TransposeMatrix(const Matrix& m);

    /**
     * @brief Check whether a square matrix is orthogonal (Q Q^T = I).
     * @param m  Square matrix.
     * @return true if orthogonal within ORTHOGONAL_EPSILON.
     * @throws std::invalid_argument if the matrix is empty or not square.
     *
     * @details Time O(n^3), Space O(n^2).
     */
    static bool IsOrthogonalMatrix(const Matrix& m);
};

#endif // ORTHOGONAL_MATRICES_HPP
