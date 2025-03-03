/**
 * @file Eigendecomposition.hpp
 * @brief Jacobi eigendecomposition for real symmetric matrices.
 *
 * The classical Jacobi method applies a sequence of Givens (plane)
 * rotations to drive off-diagonal elements toward zero.  The diagonal
 * converges to the eigenvalues; the accumulated rotations form the
 * eigenvector matrix.
 *
 * WHY JACOBI FOR SYMMETRIC MATRICES?
 *   The Spectral Theorem guarantees real symmetric matrices have:
 *     - All real eigenvalues
 *     - A complete set of orthogonal eigenvectors
 *   Jacobi rotations preserve symmetry at every step, and each
 *   rotation strictly reduces the off-diagonal Frobenius norm,
 *   guaranteeing convergence.
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <vector>

template <std::floating_point Scalar = double>
class Eigendecomposition {
public:
    /// Eigenvalues and column-major eigenvectors.
    struct EigenResult {
        std::vector<Scalar> Eigenvalues;
        std::vector<std::vector<Scalar>> Eigenvectors;
    };

    /// Decompose a real symmetric matrix A into A = V Λ V^T.
    ///
    /// Algorithm: Classical Jacobi — repeatedly zero the largest
    ///   off-diagonal element via a Givens rotation (similarity transform).
    ///
    /// Time:  O(n^3) per sweep × typically O(n) sweeps → O(n^4) worst case.
    ///        Each sweep scans all n(n-1)/2 upper-triangle pairs.
    /// Space: O(n^2) for the working copy and eigenvector matrix.
    ///
    /// Faster alternative: Householder tridiagonalization O(2n^3/3)
    ///   then implicit QR on the tridiagonal form O(n^2) → O(n^3) overall.
    ///   This is what LAPACK dsyev uses.
    ///   Jacobi is simpler, more numerically stable for small dense
    ///   matrices, and inherently parallelizable.
    ///
    /// @throws std::invalid_argument if the matrix is not square or symmetric.
    static EigenResult Decompose(
        const std::vector<std::vector<Scalar>>& symmetric_matrix);

private:
    static constexpr Scalar kTolerance = static_cast<Scalar>(1e-10);
    static constexpr int kMaxIterations = 1000;

    /// Check whether the matrix is square and symmetric.
    /// Time: O(n^2), Space: O(1).
    static bool IsSymmetric(
        const std::vector<std::vector<Scalar>>& matrix);

    /// Create an n×n identity matrix.
    /// Time: O(n^2), Space: O(n^2).
    static std::vector<std::vector<Scalar>> CreateIdentityMatrix(
        std::size_t n);
};
