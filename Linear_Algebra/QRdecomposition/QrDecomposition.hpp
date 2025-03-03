/**
 * @file QrDecomposition.hpp
 * @brief QR decomposition via the Modified Gram-Schmidt process (declarations).
 *
 * @details
 * Decomposes a matrix A into the product Q * R where:
 *   Q = orthonormal matrix  (columns are unit-length and perpendicular)
 *   R = upper-triangular matrix
 *
 * Modified Gram-Schmidt (MGS) is used instead of Classical Gram-Schmidt
 * because MGS re-orthogonalizes against the *updated* projection at
 * each step, giving much better numerical stability.
 *
 * Complexity:
 *   Decompose : O(m * n^2) time,  O(m * n) space
 *   where m = rows, n = columns of the input matrix.
 */

#ifndef QR_DECOMPOSITION_HPP
#define QR_DECOMPOSITION_HPP

#include <cstddef>
#include <vector>

/**
 * @class QrDecomposition
 * @brief QR decomposition using the Modified Gram-Schmidt process.
 *
 * @tparam Scalar  Floating-point type (float, double, long double).
 */
template <typename Scalar = double>
class QrDecomposition {
public:
    /**
     * @brief Result of a QR decomposition.
     */
    struct QrResult {
        std::vector<std::vector<Scalar>> Q;  ///< Orthonormal matrix (m x n).
        std::vector<std::vector<Scalar>> R;  ///< Upper-triangular matrix (n x n).
    };

    /**
     * @brief Perform the QR decomposition on an m x n matrix (m >= n).
     * @param input_matrix  The m x n input matrix (full column rank required).
     * @return QrResult containing Q and R such that A = Q * R.
     * @throws std::invalid_argument if the matrix is empty.
     * @throws std::runtime_error if columns are linearly dependent.
     */
    static QrResult Decompose(
        const std::vector<std::vector<Scalar>>& input_matrix);

private:
    static constexpr Scalar kEpsilon = static_cast<Scalar>(1e-10);

    /** @brief Compute the dot product of two vectors. */
    static Scalar ComputeDotProduct(const std::vector<Scalar>& a,
                                    const std::vector<Scalar>& b);

    /** @brief Compute the Euclidean norm (length) of a vector. */
    static Scalar ComputeNorm(const std::vector<Scalar>& v);
};

#endif // QR_DECOMPOSITION_HPP
