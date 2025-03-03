/**
 * @file EigendecompositionGeneral.cpp
 * @brief Implementation of QR-algorithm eigendecomposition with Householder
 *        reflections and Rayleigh quotient shift.
 */

#include "EigendecompositionGeneral.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

// ---------------------------------------------------------------------------
// DotProduct
// ---------------------------------------------------------------------------
// Time: O(n), Space: O(1).

template <std::floating_point Scalar>
Scalar EigendecompositionGeneral<Scalar>::DotProduct(
    const std::vector<Scalar>& a, const std::vector<Scalar>& b) {

    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vector sizes do not match for dot product.");
    }
    Scalar sum = static_cast<Scalar>(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ---------------------------------------------------------------------------
// VectorNorm
// ---------------------------------------------------------------------------
// Time: O(n), Space: O(1).

template <std::floating_point Scalar>
Scalar EigendecompositionGeneral<Scalar>::VectorNorm(
    const std::vector<Scalar>& v) {
    return std::sqrt(DotProduct(v, v));
}

// ---------------------------------------------------------------------------
// IdentityMatrix
// ---------------------------------------------------------------------------
// Time: O(n^2), Space: O(n^2).

template <std::floating_point Scalar>
std::vector<std::vector<Scalar>>
EigendecompositionGeneral<Scalar>::IdentityMatrix(std::size_t n) {
    std::vector<std::vector<Scalar>> id(
        n, std::vector<Scalar>(n, static_cast<Scalar>(0)));
    for (std::size_t i = 0; i < n; ++i) {
        id[i][i] = static_cast<Scalar>(1);
    }
    return id;
}

// ---------------------------------------------------------------------------
// MultiplyMatrices
// ---------------------------------------------------------------------------
// Time: O(m*n*p), Space: O(m*p).
// Faster: Strassen O(n^2.807) or BLAS-3 dgemm with cache tiling.

template <std::floating_point Scalar>
std::vector<std::vector<Scalar>>
EigendecompositionGeneral<Scalar>::MultiplyMatrices(
    const std::vector<std::vector<Scalar>>& a,
    const std::vector<std::vector<Scalar>>& b) {

    if (a.empty() || b.empty() || a[0].size() != b.size()) {
        throw std::invalid_argument(
            "Incompatible dimensions for matrix multiplication.");
    }
    std::size_t m = a.size();
    std::size_t n = a[0].size();
    std::size_t p = b[0].size();
    std::vector<std::vector<Scalar>> c(
        m, std::vector<Scalar>(p, static_cast<Scalar>(0)));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            Scalar a_ik = a[i][k];
            for (std::size_t j = 0; j < p; ++j) {
                c[i][j] += a_ik * b[k][j];
            }
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// QrDecomposition — Householder reflections
// ---------------------------------------------------------------------------
//
// OVERVIEW
//   For each column j, we compute a Householder reflector H_j that zeros
//   all sub-diagonal entries in that column:
//
//     1. Extract the sub-column x = R[j:n, j].
//     2. Choose alpha = -sign(x_0) * ||x||.
//        WHY sign choice: avoids catastrophic cancellation in v = x - alpha*e_1.
//     3. Form v = x - alpha*e_1 and normalize: v = v / ||v||.
//     4. Apply H_j = I - 2vv^T to the trailing sub-matrix:
//        R[j:n, j:n] -= 2 * v * (v^T * R[j:n, j:n])
//     5. Accumulate: Q[:, j:n] -= 2 * (Q[:, j:n] * v) * v^T
//
//   After processing all columns: A = Q*R with Q orthogonal, R upper-triangular.
//
// WHY HOUSEHOLDER OVER GRAM-SCHMIDT
//   Gram-Schmidt computes Q column-by-column via projections.  In finite
//   precision, the computed Q loses orthogonality — successive projections
//   accumulate rounding errors.  Modified Gram-Schmidt helps but is still
//   not backward-stable.
//
//   Householder reflections are backward-stable: the computed Q*R equals
//   A + E where ||E|| is O(eps * ||A||).  The orthogonality of Q is
//   maintained to machine precision.
//
// Time:  O(2n^3/3), Space: O(n^2).
// Improvement: store reflectors in compact (WY) form for BLAS-3 block
//   operations (LAPACK dgeqrf).

template <std::floating_point Scalar>
std::pair<std::vector<std::vector<Scalar>>,
          std::vector<std::vector<Scalar>>>
EigendecompositionGeneral<Scalar>::QrDecomposition(
    const std::vector<std::vector<Scalar>>& a) {

    std::size_t n = a.size();
    for (const auto& row : a) {
        if (row.size() != n) {
            throw std::invalid_argument(
                "Matrix must be square for QR decomposition.");
        }
    }

    // Trivial case: 0x0 or 1x1 matrix.
    if (n <= 1) {
        return {IdentityMatrix(n), a};
    }

    // R starts as a copy of A; will be reduced to upper triangular.
    auto r = a;
    // Q starts as identity; accumulates Householder reflections.
    auto q = IdentityMatrix(n);

    for (std::size_t j = 0; j < n - 1; ++j) {
        std::size_t m = n - j;  // length of the sub-column

        // --- Extract sub-column x = R[j:n, j] ---
        std::vector<Scalar> x(m);
        for (std::size_t i = 0; i < m; ++i) {
            x[i] = r[j + i][j];
        }

        // Compute ||x||.
        Scalar norm_x = static_cast<Scalar>(0);
        for (std::size_t i = 0; i < m; ++i) {
            norm_x += x[i] * x[i];
        }
        norm_x = std::sqrt(norm_x);

        // Skip if the sub-column is already zero.
        if (norm_x < kTolerance) continue;

        // --- Choose alpha = -sign(x_0) * ||x|| ---
        // WHY: This sign convention maximizes |v[0]| = |x[0] - alpha|,
        // avoiding cancellation and improving numerical stability.
        Scalar alpha = (x[0] >= static_cast<Scalar>(0)) ? -norm_x : norm_x;

        // --- Form Householder vector v = x - alpha*e_1, then normalize ---
        // WHY: H = I - 2vv^T maps x to alpha*e_1, zeroing sub-diagonal entries.
        std::vector<Scalar> v = x;
        v[0] -= alpha;

        Scalar norm_v = static_cast<Scalar>(0);
        for (std::size_t i = 0; i < m; ++i) {
            norm_v += v[i] * v[i];
        }
        norm_v = std::sqrt(norm_v);
        if (norm_v < std::numeric_limits<Scalar>::epsilon()) continue;
        for (std::size_t i = 0; i < m; ++i) {
            v[i] /= norm_v;
        }

        // --- Apply reflector to R: R[j:n, j:n] -= 2*v*(v^T * R[j:n, j:n]) ---
        // WHY: This transforms column j so that R[j+1:n, j] = 0.
        // Subsequent columns j+1..n-1 are also updated to maintain A = Q*R.
        for (std::size_t col = j; col < n; ++col) {
            Scalar dot = static_cast<Scalar>(0);
            for (std::size_t i = 0; i < m; ++i) {
                dot += v[i] * r[j + i][col];
            }
            for (std::size_t i = 0; i < m; ++i) {
                r[j + i][col] -= static_cast<Scalar>(2) * v[i] * dot;
            }
        }

        // --- Accumulate Q: Q[:, j:n] -= 2*(Q[:, j:n]*v)*v^T ---
        // WHY: Q = H_1 * H_2 * ... * H_{n-1}.  We build Q incrementally
        // by right-multiplying each reflector: Q_new = Q_old * H_j.
        for (std::size_t row = 0; row < n; ++row) {
            Scalar dot = static_cast<Scalar>(0);
            for (std::size_t i = 0; i < m; ++i) {
                dot += q[row][j + i] * v[i];
            }
            for (std::size_t i = 0; i < m; ++i) {
                q[row][j + i] -= static_cast<Scalar>(2) * dot * v[i];
            }
        }
    }

    return {q, r};
}

// ---------------------------------------------------------------------------
// OffDiagonalFrobeniusNorm
// ---------------------------------------------------------------------------
// Time: O(n^2), Space: O(1).

template <std::floating_point Scalar>
Scalar EigendecompositionGeneral<Scalar>::OffDiagonalFrobeniusNorm(
    const std::vector<std::vector<Scalar>>& a) {

    std::size_t n = a.size();
    Scalar sum = static_cast<Scalar>(0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j) {
                sum += a[i][j] * a[i][j];
            }
        }
    }
    return std::sqrt(sum);
}

// ---------------------------------------------------------------------------
// Decompose — QR algorithm with Rayleigh quotient shift
// ---------------------------------------------------------------------------
//
// OVERVIEW
//   The QR algorithm iterates A_k -> A_{k+1} where:
//     A_k - sigma*I = Q*R        (QR decompose the shifted matrix)
//     A_{k+1} = R*Q + sigma*I    (reverse multiply and un-shift)
//
//   This is a similarity transform: A_{k+1} = Q^T * A_k * Q.
//   Eigenvalues are preserved.  The matrix converges to upper-triangular
//   (Schur) form; diagonal entries -> eigenvalues.
//
// WHY SHIFT?
//   Without shift (sigma = 0): convergence is linear, proportional to
//   |lambda_i / lambda_{i+1}|.  Slow when eigenvalues are close.
//
//   With Rayleigh shift (sigma = a[n-1][n-1]): convergence becomes
//   cubic for symmetric matrices and typically quadratic for general.
//   When sigma is close to an eigenvalue, A - sigma*I is nearly singular,
//   which drives the bottom-right subdiagonal toward zero fast (deflation).
//
// WHY ACCUMULATE Q?
//   The product V = Q_1 * Q_2 * ... converges to the eigenvector matrix.
//   After convergence, A_final = V^T * A_original * V.
//
// Time:  O(n^3) per iteration, typically O(n) iterations → O(n^4).
// Space: O(n^2).

template <std::floating_point Scalar>
typename EigendecompositionGeneral<Scalar>::GeneralEigenResult
EigendecompositionGeneral<Scalar>::Decompose(
    const std::vector<std::vector<Scalar>>& matrix) {

    auto a = matrix;
    std::size_t n = a.size();
    for (const auto& row : a) {
        if (row.size() != n) {
            throw std::invalid_argument(
                "Matrix must be square for eigendecomposition.");
        }
    }
    if (n == 0) {
        throw std::invalid_argument("Matrix must not be empty.");
    }

    auto eigenvectors = IdentityMatrix(n);

    for (int iter = 0; iter < kMaxIterations; ++iter) {
        // --- Deflation: find the active (unreduced) block ---
        // WHY: When a sub-diagonal element a[k][k-1] is negligible, the
        // eigenvalue a[k][k]..a[n-1][n-1] block has converged.  We only
        // need to reduce the remaining top-left block.  This is critical
        // for non-symmetric matrices where the Rayleigh shift based on the
        // already-converged bottom-right element would fail to reduce the
        // upper block.
        std::size_t active = n;
        while (active > 1 &&
               std::fabs(a[active - 1][active - 2]) < kTolerance) {
            --active;
        }
        if (active <= 1) break;  // All sub-diagonals are zero — converged.

        // --- Wilkinson shift: eigenvalue of bottom 2x2 closest to a[p-1][p-1] ---
        // WHY Wilkinson over simple Rayleigh?
        //   Rayleigh (sigma = a[p-1][p-1]) can oscillate for non-symmetric
        //   matrices when the 2x2 block has complex eigenvalues.
        //   Wilkinson always converges and gives cubic convergence for
        //   symmetric matrices, quadratic for general.
        Scalar shift;
        {
            std::size_t p = active;
            Scalar d = (a[p - 2][p - 2] - a[p - 1][p - 1]) /
                       static_cast<Scalar>(2);
            Scalar mu = a[p - 1][p - 2] * a[p - 1][p - 2];
            if (d == static_cast<Scalar>(0)) {
                shift = a[p - 1][p - 1] - std::sqrt(mu);
            } else {
                Scalar sign_d = (d > static_cast<Scalar>(0))
                                    ? static_cast<Scalar>(1)
                                    : static_cast<Scalar>(-1);
                shift = a[p - 1][p - 1] -
                        mu / (d + sign_d * std::sqrt(d * d + mu));
            }
        }

        // Shift: A <- A - sigma*I
        for (std::size_t i = 0; i < n; ++i) {
            a[i][i] -= shift;
        }

        // QR decompose the shifted matrix using Householder reflections.
        auto [q, r] = QrDecomposition(a);

        // Form next iterate: A <- R*Q + sigma*I  (similarity transform).
        a = MultiplyMatrices(r, q);
        for (std::size_t i = 0; i < n; ++i) {
            a[i][i] += shift;
        }

        // Accumulate eigenvectors: V <- V * Q.
        eigenvectors = MultiplyMatrices(eigenvectors, q);
    }

    // Read eigenvalues from the diagonal of the (now nearly triangular) matrix.
    std::vector<Scalar> eigenvalues(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = a[i][i];
    }

    // --- Back-substitution: extract true eigenvectors from Schur form ---
    // WHY: The accumulated Q product gives Schur vectors, not eigenvectors.
    //   For symmetric matrices, Schur form is diagonal, so Schur vectors
    //   ARE eigenvectors.  For non-symmetric matrices, the Schur form T is
    //   upper-triangular.  We solve (T - lambda_i * I) x = 0 for each
    //   eigenvalue by back-substitution, then transform: v = V * x.
    //   This is what LAPACK's dtrevc does.
    //
    // Time: O(n^3), dominated by the QR iteration.

    auto schur_eigvecs = IdentityMatrix(n);
    for (std::size_t col = 0; col < n; ++col) {
        Scalar lambda = a[col][col];
        schur_eigvecs[col][col] = static_cast<Scalar>(1);
        for (std::size_t k = col; k-- > 0;) {
            Scalar sum = static_cast<Scalar>(0);
            for (std::size_t j = k + 1; j <= col; ++j) {
                sum += a[k][j] * schur_eigvecs[j][col];
            }
            Scalar denom = a[k][k] - lambda;
            if (std::fabs(denom) < kTolerance) {
                denom = kTolerance;
            }
            schur_eigvecs[k][col] = -sum / denom;
        }
    }

    // Transform back: eigenvectors of A = V * eigenvectors of T.
    eigenvectors = MultiplyMatrices(eigenvectors, schur_eigvecs);

    // Normalize each eigenvector to unit length.
    for (std::size_t col = 0; col < n; ++col) {
        Scalar norm = static_cast<Scalar>(0);
        for (std::size_t i = 0; i < n; ++i) {
            norm += eigenvectors[i][col] * eigenvectors[i][col];
        }
        norm = std::sqrt(norm);
        if (norm > kTolerance) {
            for (std::size_t i = 0; i < n; ++i) {
                eigenvectors[i][col] /= norm;
            }
        }
    }

    return {eigenvalues, eigenvectors};
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template class EigendecompositionGeneral<float>;
template class EigendecompositionGeneral<double>;
template class EigendecompositionGeneral<long double>;
