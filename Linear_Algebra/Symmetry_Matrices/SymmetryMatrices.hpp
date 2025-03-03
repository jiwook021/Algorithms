/**
 * @file SymmetryMatrices.hpp
 * @brief Properties and operations on symmetric matrices.
 * @details Provides a SymmetryMatrix class that wraps a dense square matrix
 *          and exposes symmetry checks, trace, determinant (up to 3x3),
 *          Frobenius norm, positive-definiteness test, eigenvalue
 *          approximation via power iteration, and matrix arithmetic.
 */

#ifndef SYMMETRY_MATRICES_HPP
#define SYMMETRY_MATRICES_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

/**
 * @brief Symmetric matrix with property queries and basic operations.
 *
 * @details Internally stores a dense n x n matrix.  Construction from raw
 *          data does NOT enforce symmetry; use IsSymmetric() to verify.
 *          The random constructor guarantees A[i][j] == A[j][i].
 */
class SymmetryMatrix {
public:
    /**
     * @brief Construct a symmetric matrix from explicit data.
     * @param data  n x n matrix data.
     * @throws std::invalid_argument if data is empty or not square.
     */
    explicit SymmetryMatrix(const std::vector<std::vector<double>>& data);

    /**
     * @brief Construct a random symmetric matrix of given size.
     * @param n  Dimension (must be > 0).
     * @param seed  RNG seed (0 = non-deterministic).
     * @throws std::invalid_argument if n == 0.
     */
    explicit SymmetryMatrix(std::size_t n, unsigned int seed = 0);

    /**
     * @brief Check whether the matrix is symmetric within epsilon.
     * @return true if A[i][j] == A[j][i] for all i, j.
     * @details Time O(n^2), Space O(1).
     */
    bool IsSymmetric() const;

    /**
     * @brief Compute the trace (sum of diagonal elements).
     * @return Trace value.
     * @details Time O(n), Space O(1).
     */
    double Trace() const;

    /**
     * @brief Compute the determinant (only for n <= 3).
     * @return Determinant value.
     * @throws std::logic_error for n > 3.
     * @details Time O(1) for n <= 2, O(1) for n == 3.
     */
    double Determinant() const;

    /**
     * @brief Compute the Frobenius norm.
     * @return sqrt(sum A[i][j]^2).
     * @details Time O(n^2), Space O(1).
     */
    double FrobeniusNorm() const;

    /**
     * @brief Approximate the largest eigenvalue via power iteration.
     * @param iterations  Number of iterations.
     * @return Approximate dominant eigenvalue (Rayleigh quotient).
     * @details Time O(n^2 * iterations), Space O(n).
     */
    double ApproximateLargestEigenvalue(std::size_t iterations = 100) const;

    /**
     * @brief Check positive-definiteness (n <= 3, Sylvester's criterion).
     * @return true if all leading principal minors are positive.
     * @throws std::logic_error for n > 3.
     */
    bool IsPositiveDefinite() const;

    /**
     * @brief Matrix addition (symmetric + symmetric = symmetric).
     * @param other  Matrix of the same dimension.
     * @return Sum matrix.
     * @throws std::invalid_argument if dimensions differ.
     * @details Time O(n^2), Space O(n^2).
     */
    SymmetryMatrix Add(const SymmetryMatrix& other) const;

    /**
     * @brief Matrix multiplication.
     * @param other  Matrix of the same dimension.
     * @return Product as raw data (may not be symmetric).
     * @throws std::invalid_argument if dimensions differ.
     * @details Time O(n^3), Space O(n^2).
     */
    std::vector<std::vector<double>> Multiply(
        const SymmetryMatrix& other) const;

    /**
     * @brief Check whether the product A*B is symmetric.
     * @details Time O(n^3), Space O(n^2).
     */
    bool IsProductSymmetric(const SymmetryMatrix& other) const;

    /**
     * @brief Check whether A*B == B*A (commutativity).
     * @details Time O(n^3), Space O(n^2).
     */
    bool IsCommutative(const SymmetryMatrix& other) const;

    /** @brief Element access. */
    double GetValue(std::size_t i, std::size_t j) const;

    /** @brief Set element. */
    void SetValue(std::size_t i, std::size_t j, double value);

    /** @brief Matrix dimension. */
    std::size_t GetSize() const;

    /** @brief Print to stdout. */
    void Print() const;

private:
    std::vector<std::vector<double>> matrix_;
    std::size_t size_;

    static constexpr double EPSILON = 1e-10;
};

/**
 * @brief Create an n x n identity SymmetryMatrix.
 * @param n  Dimension.
 * @return Identity matrix.
 */
SymmetryMatrix CreateIdentityMatrix(std::size_t n);

#endif // SYMMETRY_MATRICES_HPP
