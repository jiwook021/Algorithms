/**
 * @file EigendecompositionGeneral.hpp
 * @brief QR-algorithm eigendecomposition for general (non-symmetric) matrices.
 *
 * Uses the QR algorithm with Householder reflections and Rayleigh
 * quotient shift.  Each iteration performs A = QR, then forms A' = RQ,
 * which is a similarity transform (same eigenvalues, more upper-triangular).
 * The shift accelerates convergence from linear to cubic.
 *
 * WHY QR FOR NON-SYMMETRIC MATRICES?
 *   The Jacobi method requires symmetry.  The QR algorithm works for
 *   any square matrix with real eigenvalues by iteratively driving it
 *   toward upper-triangular (Schur) form.
 *
 * WHY HOUSEHOLDER OVER GRAM-SCHMIDT?
 *   Classical Gram-Schmidt suffers from loss of orthogonality in finite
 *   precision.  Householder reflections are backward-stable: the
 *   computed Q is orthogonal to machine precision.  LAPACK's dgeqrf
 *   uses Householder for this reason.
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

template <std::floating_point Scalar = double>
class EigendecompositionGeneral {
public:
    /// Eigenvalues and accumulated eigenvector columns.
    struct GeneralEigenResult {
        std::vector<Scalar> Eigenvalues;
        std::vector<std::vector<Scalar>> Eigenvectors;
    };

    /// Decompose a square matrix into eigenvalues and eigenvectors.
    ///
    /// Algorithm: QR iteration with Householder QR and Rayleigh shift.
    ///   1. Deflate: scan from bottom for converged eigenvalues
    ///   2. Choose Wilkinson shift σ from active 2×2 sub-block
    ///   3. QR decompose A - σI = QR      (Householder reflections)
    ///   4. Form next iterate A' = RQ + σI (similarity transform)
    ///   5. Accumulate eigenvectors V = V · Q
    ///   6. Repeat until sub-diagonal → 0, then back-substitute
    ///
    /// Time:  O(n^3) per iteration × O(n) iterations typical → O(n^4).
    ///        Wilkinson shift gives cubic convergence for symmetric matrices
    ///        and quadratic for general — far fewer iterations in practice.
    /// Space: O(n^2) for working matrices.
    ///
    /// Faster alternative: Reduce to upper Hessenberg form first via
    ///   Householder O(10n^3/3), then each QR step costs O(n^2) instead
    ///   of O(n^3), giving O(n^3) total.  This is what LAPACK dgeev uses.
    ///
    /// @throws std::invalid_argument if the matrix is not square or empty.
    static GeneralEigenResult Decompose(
        const std::vector<std::vector<Scalar>>& matrix);

    // ----- Utility functions (public for testability) -----

    /// Dot product of two equal-length vectors.
    /// Time: O(n), Space: O(1).
    static Scalar DotProduct(const std::vector<Scalar>& a,
                             const std::vector<Scalar>& b);

    /// Euclidean (L2) norm.
    /// Time: O(n), Space: O(1).
    static Scalar VectorNorm(const std::vector<Scalar>& v);

    /// Matrix multiplication C = A · B.
    /// Time: O(m·n·p), Space: O(m·p).
    /// Faster: Strassen O(n^2.807) or BLAS dgemm with cache tiling.
    static std::vector<std::vector<Scalar>> MultiplyMatrices(
        const std::vector<std::vector<Scalar>>& a,
        const std::vector<std::vector<Scalar>>& b);

    /// QR decomposition via Householder reflections.
    /// Returns (Q, R) such that A = Q·R, Q orthogonal, R upper triangular.
    ///
    /// Algorithm: For each column j, compute a Householder reflector
    ///   H_j = I - 2vv^T that zeros sub-diagonal entries in column j.
    ///   Apply H_j to the trailing sub-matrix of R, and accumulate
    ///   Q = Q · H_j.
    ///
    /// Time:  O(2n^3/3), Space: O(n^2).
    /// Improvement: Store reflectors in compact (WY) form to exploit
    ///   BLAS-3 block operations — used in LAPACK dgeqrf.
    static std::pair<std::vector<std::vector<Scalar>>,
                     std::vector<std::vector<Scalar>>>
    QrDecomposition(const std::vector<std::vector<Scalar>>& a);

    /// Frobenius norm of off-diagonal elements (convergence criterion).
    /// Time: O(n^2), Space: O(1).
    static Scalar OffDiagonalFrobeniusNorm(
        const std::vector<std::vector<Scalar>>& a);

    /// Create an n×n identity matrix.
    /// Time: O(n^2), Space: O(n^2).
    static std::vector<std::vector<Scalar>> IdentityMatrix(std::size_t n);

private:
    static constexpr Scalar kTolerance = static_cast<Scalar>(1e-10);
    static constexpr int kMaxIterations = 1000;
};
