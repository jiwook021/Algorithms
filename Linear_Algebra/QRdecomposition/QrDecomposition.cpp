/**
 * @file QrDecomposition.cpp
 * @brief Implementation of QR decomposition (Modified Gram-Schmidt).
 */

#include "QrDecomposition.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Helper: dot product
// ---------------------------------------------------------------------------

template <typename Scalar>
Scalar QrDecomposition<Scalar>::ComputeDotProduct(
        const std::vector<Scalar>& a, const std::vector<Scalar>& b) {
    //
    // dot(a, b) = a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]
    //
    // Measures how much two vectors point in the same direction.
    //   dot = 0  ->  perpendicular (orthogonal)
    //   dot > 0  ->  same general direction
    //   dot < 0  ->  opposite direction
    //
    Scalar sum = static_cast<Scalar>(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Helper: Euclidean norm (length of a vector)
// ---------------------------------------------------------------------------

template <typename Scalar>
Scalar QrDecomposition<Scalar>::ComputeNorm(const std::vector<Scalar>& v) {
    //
    // ||v|| = sqrt(v[0]^2 + v[1]^2 + ... + v[n-1]^2)
    //
    // This is the "length" of the vector.  A unit vector has norm = 1.
    //
    return std::sqrt(ComputeDotProduct(v, v));
}

// ---------------------------------------------------------------------------
// Modified Gram-Schmidt  (the main algorithm)
// ---------------------------------------------------------------------------
//
// Goal: Factor matrix A into Q * R where
//   Q has orthonormal columns (unit-length and mutually perpendicular)
//   R is upper-triangular
//
// Why "Modified"?
//   Classical Gram-Schmidt: for each new column, subtract projections
//   onto ALL previous columns using the ORIGINAL vectors.
//
//   Modified Gram-Schmidt: for each column k, immediately update all
//   REMAINING columns (k+1 .. n-1) using the freshly-normalized q_k.
//
//   The difference is subtle but crucial for numerical stability:
//   - Classical GS: orthogonality error ~ eps * cond(A)
//   - Modified  GS: orthogonality error ~ eps   (machine precision)
//
//   where cond(A) is the condition number of A.  For ill-conditioned
//   matrices (cond >> 1), Classical GS can produce vectors that are
//   far from orthogonal, while Modified GS stays accurate.
//
// ---------------------------------------------------------------------------

template <typename Scalar>
auto QrDecomposition<Scalar>::Decompose(
        const std::vector<std::vector<Scalar>>& input_matrix) -> QrResult {
    if (input_matrix.empty() || input_matrix[0].empty()) {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    const std::size_t num_rows = input_matrix.size();
    const std::size_t num_cols = input_matrix[0].size();

    // ------------------------------------------------------------------
    // Step 0: Copy columns of A into working vectors.
    //
    // We store columns (not rows) because Gram-Schmidt processes
    // one column at a time.  q_cols[j] = j-th column of A.
    //
    //   A = [ 1  1  0 ]     q_cols[0] = [1, 1, 0]
    //       [ 1  0  1 ]     q_cols[1] = [1, 0, 1]
    //       [ 0  1  1 ]     q_cols[2] = [0, 1, 1]
    // ------------------------------------------------------------------

    std::vector<std::vector<Scalar>> q_cols(
        num_cols, std::vector<Scalar>(num_rows, static_cast<Scalar>(0)));

    for (std::size_t j = 0; j < num_cols; ++j) {
        for (std::size_t i = 0; i < num_rows; ++i) {
            q_cols[j][i] = input_matrix[i][j];
        }
    }

    // R is n x n upper-triangular (filled as we go).
    std::vector<std::vector<Scalar>> r(
        num_cols, std::vector<Scalar>(num_cols, static_cast<Scalar>(0)));

    // ------------------------------------------------------------------
    // Modified Gram-Schmidt loop
    //
    // For each column k:
    //   1. Compute its norm -> R[k][k]
    //   2. Normalize it -> q_k becomes a unit vector
    //   3. For every later column j (j > k):
    //      a) Compute how much j overlaps with q_k  -> R[k][j]
    //      b) Subtract that overlap from column j
    //
    // After processing column k, all later columns have been
    // stripped of their component in the q_k direction.
    //
    //   Example (2D):
    //     v1 = [3, 4]     ||v1|| = 5
    //     q1 = [0.6, 0.8] (normalized)
    //
    //     v2 = [1, 2]
    //     overlap = dot(q1, v2) = 0.6*1 + 0.8*2 = 2.2
    //     v2 = v2 - 2.2 * q1 = [1 - 1.32, 2 - 1.76] = [-0.32, 0.24]
    //     q2 = normalize(v2) = [-0.8, 0.6]
    //
    //     Now dot(q1, q2) = 0.6*(-0.8) + 0.8*0.6 = 0  (orthogonal!)
    // ------------------------------------------------------------------

    for (std::size_t k = 0; k < num_cols; ++k) {

        // 1. Norm of the current column = diagonal entry of R.
        r[k][k] = ComputeNorm(q_cols[k]);

        // If the norm is ~0, the column is linearly dependent on
        // the previous columns (it got reduced to nothing).
        if (r[k][k] < kEpsilon) {
            throw std::runtime_error(
                "Matrix columns are linearly dependent.");
        }

        // 2. Normalize: divide every element by the norm.
        //    After this, q_cols[k] is a unit vector (length = 1).
        for (std::size_t i = 0; i < num_rows; ++i) {
            q_cols[k][i] /= r[k][k];
        }

        // 3. Orthogonalize every remaining column against q_k.
        //    This is the "Modified" part -- we use the freshly
        //    normalized q_k to update the later columns immediately.
        for (std::size_t j = k + 1; j < num_cols; ++j) {
            // How much does column j overlap with q_k?
            r[k][j] = ComputeDotProduct(q_cols[k], q_cols[j]);

            // Remove that overlap.
            for (std::size_t i = 0; i < num_rows; ++i) {
                q_cols[j][i] -= r[k][j] * q_cols[k][i];
            }
        }
    }

    // ------------------------------------------------------------------
    // Convert column-major q_cols back to row-major Q.
    //
    //   q_cols[0] = [0.6, 0.8]    Q = [ 0.6  -0.8 ]
    //   q_cols[1] = [-0.8, 0.6]       [ 0.8   0.6 ]
    // ------------------------------------------------------------------

    std::vector<std::vector<Scalar>> q(
        num_rows, std::vector<Scalar>(num_cols, static_cast<Scalar>(0)));
    for (std::size_t j = 0; j < num_cols; ++j) {
        for (std::size_t i = 0; i < num_rows; ++i) {
            q[i][j] = q_cols[j][i];
        }
    }

    return {q, r};
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template class QrDecomposition<float>;
template class QrDecomposition<double>;
template class QrDecomposition<long double>;
