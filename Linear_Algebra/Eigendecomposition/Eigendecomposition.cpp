/**
 * @file Eigendecomposition.cpp
 * @brief Implementation of the Jacobi eigendecomposition algorithm.
 */

#include "Eigendecomposition.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

// ---------------------------------------------------------------------------
// IsSymmetric
// ---------------------------------------------------------------------------
// Time: O(n^2) — checks every upper-triangle pair.
// Space: O(1).

template <std::floating_point Scalar>
bool Eigendecomposition<Scalar>::IsSymmetric(
    const std::vector<std::vector<Scalar>>& matrix) {

    const auto n = matrix.size();
    if (n == 0) return false;
    for (std::size_t i = 0; i < n; ++i) {
        if (matrix[i].size() != n) return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::fabs(matrix[i][j] - matrix[j][i]) > kTolerance) {
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CreateIdentityMatrix
// ---------------------------------------------------------------------------
// Time: O(n^2), Space: O(n^2).

template <std::floating_point Scalar>
std::vector<std::vector<Scalar>>
Eigendecomposition<Scalar>::CreateIdentityMatrix(std::size_t n) {
    std::vector<std::vector<Scalar>> identity(
        n, std::vector<Scalar>(n, static_cast<Scalar>(0)));
    for (std::size_t i = 0; i < n; ++i) {
        identity[i][i] = static_cast<Scalar>(1);
    }
    return identity;
}

// ---------------------------------------------------------------------------
// Decompose — Jacobi eigenvalue algorithm
// ---------------------------------------------------------------------------
//
// OVERVIEW
//   A Givens rotation G(p,q,theta) is an orthogonal matrix that differs
//   from the identity only in rows/columns p and q.  The similarity
//   transform A' = G^T A G preserves eigenvalues (similar matrices share
//   spectra) and zeros out the (p,q) and (q,p) off-diagonal entries.
//
//   Over many sweeps, ALL off-diagonal elements shrink toward zero.
//   The diagonal converges to the eigenvalues, and the accumulated
//   product of rotations G_1 G_2 ... converges to the eigenvector matrix.
//
// WHY SYMMETRIC IS SPECIAL
//   - Symmetry is preserved by orthogonal similarity transforms:
//     if A is symmetric, G^T A G is also symmetric.
//   - Each rotation strictly reduces the sum of squared off-diagonal
//     elements, guaranteeing convergence.
//   - The Spectral Theorem guarantees all eigenvalues are real and
//     eigenvectors can be chosen orthonormal.
//
// Time:  O(n^3) per sweep × typically O(n) sweeps → O(n^4) worst case.
// Space: O(n^2).

template <std::floating_point Scalar>
typename Eigendecomposition<Scalar>::EigenResult
Eigendecomposition<Scalar>::Decompose(
    const std::vector<std::vector<Scalar>>& symmetric_matrix) {

    if (!IsSymmetric(symmetric_matrix)) {
        throw std::invalid_argument(
            "Input matrix must be square and symmetric.");
    }

    // Work on a mutable copy — will converge to a diagonal matrix.
    auto matrix = symmetric_matrix;
    const auto n = matrix.size();
    auto eigenvectors = CreateIdentityMatrix(n);

    for (int iter = 0; iter < kMaxIterations; ++iter) {

        // ----- Step 1: Find the largest off-diagonal |a[p][q]| -----
        // WHY: Zeroing the largest element gives the fastest reduction
        // in the off-diagonal Frobenius norm per rotation.
        std::size_t p = 0, q = 1;
        Scalar max_off_diag = static_cast<Scalar>(0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                Scalar abs_val = std::fabs(matrix[i][j]);
                if (abs_val > max_off_diag) {
                    max_off_diag = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        // Converged: all off-diagonal elements are below tolerance.
        if (max_off_diag < kTolerance) break;

        // ----- Step 2: Compute rotation angle theta -----
        // WHY: We choose theta so the (p,q) element becomes exactly zero.
        // From the condition a'[p][q] = 0 after rotation:
        //   (a[q][q] - a[p][p]) sin(2*theta)/2 + a[p][q] cos(2*theta) = 0
        //   => tan(2*theta) = 2*a[p][q] / (a[q][q] - a[p][p])
        Scalar theta;
        if (std::fabs(matrix[q][q] - matrix[p][p]) < kTolerance) {
            // Diagonal elements are equal => theta = +/- pi/4.
            theta = (matrix[p][q] > static_cast<Scalar>(0))
                        ? static_cast<Scalar>(std::numbers::pi / 4.0)
                        : static_cast<Scalar>(-std::numbers::pi / 4.0);
        } else {
            theta = static_cast<Scalar>(0.5) *
                    std::atan2(static_cast<Scalar>(2) * matrix[p][q],
                               matrix[q][q] - matrix[p][p]);
        }
        Scalar c = std::cos(theta);
        Scalar s = std::sin(theta);

        // ----- Step 3: Apply the Givens rotation A' = G^T A G -----
        // WHY: This is a similarity transform, so A' has the same
        // eigenvalues as A, but with a'[p][q] = a'[q][p] = 0.
        // Only rows/columns p and q change.
        Scalar a_pp = matrix[p][p];
        Scalar a_qq = matrix[q][q];
        Scalar a_pq = matrix[p][q];

        // New diagonal entries (rotated).
        matrix[p][p] = c * c * a_pp
                     - static_cast<Scalar>(2) * s * c * a_pq
                     + s * s * a_qq;
        matrix[q][q] = s * s * a_pp
                     + static_cast<Scalar>(2) * s * c * a_pq
                     + c * c * a_qq;

        // Target: zero the off-diagonal pair.
        matrix[p][q] = matrix[q][p] = static_cast<Scalar>(0);

        // Update rows/columns p and q for all other indices.
        for (std::size_t i = 0; i < n; ++i) {
            if (i != p && i != q) {
                Scalar a_ip = matrix[i][p];
                Scalar a_iq = matrix[i][q];
                matrix[i][p] = matrix[p][i] = c * a_ip - s * a_iq;
                matrix[i][q] = matrix[q][i] = s * a_ip + c * a_iq;
            }
        }

        // ----- Step 4: Accumulate eigenvectors -----
        // WHY: The eigenvectors of the original matrix are the product
        // of all Givens rotations: V = G_1 * G_2 * ... * G_k.
        // We apply each rotation to V on the right.
        for (std::size_t i = 0; i < n; ++i) {
            Scalar v_ip = eigenvectors[i][p];
            Scalar v_iq = eigenvectors[i][q];
            eigenvectors[i][p] = c * v_ip - s * v_iq;
            eigenvectors[i][q] = s * v_ip + c * v_iq;
        }
    }

    // Read eigenvalues from the (now nearly diagonal) matrix.
    std::vector<Scalar> eigenvalues(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = matrix[i][i];
    }

    return {eigenvalues, eigenvectors};
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template class Eigendecomposition<float>;
template class Eigendecomposition<double>;
template class Eigendecomposition<long double>;
