/**
 * @file SingularValueDecomposition.hpp
 * @brief Singular Value Decomposition (SVD) via power iteration.
 * @details Decomposes an m x n matrix A into A = U * Sigma * V^T where
 *          U (m x k) contains left singular vectors, V (n x k) contains
 *          right singular vectors, and Sigma is diagonal with non-negative
 *          singular values in descending order.
 *
 *          Templatized on scalar type using the C++20 std::floating_point
 *          concept so that the same algorithm works with float, double, or
 *          long double.
 *
 * Time Complexity:  O(k * iterations * m * n)  where k = min(m,n)
 * Space Complexity: O(m*n + m^2 + n^2)
 */

#ifndef SINGULAR_VALUE_DECOMPOSITION_HPP
#define SINGULAR_VALUE_DECOMPOSITION_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <vector>

/**
 * @brief SVD via deflated power iteration.
 *
 * @tparam Scalar  Floating-point type (float, double, long double).
 *                 Constrained by std::floating_point.
 */
template <std::floating_point Scalar = double>
class SingularValueDecomposition {
public:
    using Matrix = std::vector<std::vector<Scalar>>;
    using Vector = std::vector<Scalar>;

    /**
     * @brief Result of a singular value decomposition.
     */
    struct SvdResult {
        Matrix U;                ///< Left singular vectors (m x m).
        Vector SingularValues;   ///< Singular values in descending order.
        Matrix Vt;               ///< Right singular vectors transposed (n x n).
    };

    /**
     * @brief Compute the SVD of a matrix.
     * @param a  Input matrix (m x n).
     * @return SvdResult with U, SingularValues, Vt.
     * @throws std::invalid_argument if the matrix is empty or non-rectangular.
     */
    static SvdResult Decompose(const Matrix& a);

    /**
     * @brief Reconstruct A from its SVD components.
     * @param result  An SvdResult previously computed.
     * @return The reconstructed matrix A = U * Sigma * Vt.
     */
    static Matrix Reconstruct(const SvdResult& result);

    /**
     * @brief Compute the pseudoinverse A^+ = V Sigma^+ U^T.
     * @param result  SVD result.
     * @param tolerance  Threshold below which singular values are treated as zero.
     * @return Pseudoinverse matrix (n x m).
     */
    static Matrix Pseudoinverse(const SvdResult& result,
                                Scalar tolerance = static_cast<Scalar>(-1));

    /**
     * @brief Numerical rank of the matrix.
     * @param result  SVD result.
     * @param tolerance  Threshold for zero singular values.
     * @return Number of singular values above the tolerance.
     */
    static std::size_t Rank(const SvdResult& result,
                            Scalar tolerance = static_cast<Scalar>(-1));

    /**
     * @brief Condition number (ratio of largest to smallest nonzero SV).
     * @param result  SVD result.
     * @return Condition number (infinity if rank-deficient).
     */
    static Scalar ConditionNumber(const SvdResult& result);

    /**
     * @brief Frobenius norm sqrt(sum s_i^2).
     * @param result  SVD result.
     * @return Frobenius norm.
     */
    static Scalar FrobeniusNorm(const SvdResult& result);

    // --- Utility (public for testability) ---

    static Matrix Transpose(const Matrix& a);
    static Matrix MatrixMultiply(const Matrix& a, const Matrix& b);
    static Scalar DotProduct(const Vector& a, const Vector& b);

private:
    /// Convergence threshold for power iteration.
    static constexpr Scalar kEpsilon = static_cast<Scalar>(1e-10);

    /// Maximum number of power iterations per singular value.
    static constexpr int kMaxIterations = 1000;
};

// ============================================================================
// Implementation (must be in header because of template)
// ============================================================================

// ----------------------------------------------------------------------------
// Utility
// ----------------------------------------------------------------------------

template <std::floating_point Scalar>
Scalar SingularValueDecomposition<Scalar>::DotProduct(const Vector& a,
                                                      const Vector& b) {
    Scalar sum = static_cast<Scalar>(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

template <std::floating_point Scalar>
typename SingularValueDecomposition<Scalar>::Matrix
SingularValueDecomposition<Scalar>::Transpose(const Matrix& a) {
    if (a.empty()) return {};
    std::size_t m = a.size(), n = a[0].size();
    Matrix t(n, Vector(m));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            t[j][i] = a[i][j];
        }
    }
    return t;
}

template <std::floating_point Scalar>
typename SingularValueDecomposition<Scalar>::Matrix
SingularValueDecomposition<Scalar>::MatrixMultiply(const Matrix& a,
                                                    const Matrix& b) {
    if (a.empty() || b.empty()) {
        throw std::invalid_argument("Matrices cannot be empty.");
    }
    std::size_t m = a.size(), k = a[0].size(), n = b[0].size();
    if (k != b.size()) {
        throw std::invalid_argument("Dimension mismatch.");
    }
    Matrix c(m, Vector(n, static_cast<Scalar>(0)));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t l = 0; l < k; ++l) {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    return c;
}

// ----------------------------------------------------------------------------
// Core SVD computation
// ----------------------------------------------------------------------------

template <std::floating_point Scalar>
typename SingularValueDecomposition<Scalar>::SvdResult
SingularValueDecomposition<Scalar>::Decompose(const Matrix& a) {
    if (a.empty() || a[0].empty()) {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }
    std::size_t cols = a[0].size();
    for (const auto& row : a) {
        if (row.size() != cols) {
            throw std::invalid_argument("Matrix must be rectangular.");
        }
    }

    std::size_t m = a.size();
    std::size_t n = cols;
    std::size_t k = std::min(m, n);

    constexpr Scalar zero = static_cast<Scalar>(0);
    constexpr Scalar one  = static_cast<Scalar>(1);

    // Initialise U, V as identity.
    Matrix u(m, Vector(m, zero));
    Matrix v(n, Vector(n, zero));
    Vector s(k, zero);
    for (std::size_t i = 0; i < m; ++i) u[i][i] = one;
    for (std::size_t i = 0; i < n; ++i) v[i][i] = one;

    Matrix b = a;  // working copy (deflated after each SV)

    for (std::size_t idx = 0; idx < k; ++idx) {
        Vector v_vec(n, zero);
        v_vec[idx] = one;

        for (int iter = 0; iter < kMaxIterations; ++iter) {
            // u_vec = B * v_vec
            Vector u_vec(m, zero);
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    u_vec[i] += b[i][j] * v_vec[j];
                }
            }

            Scalar sigma = std::sqrt(DotProduct(u_vec, u_vec));
            if (sigma < kEpsilon) break;
            for (auto& val : u_vec) val /= sigma;

            // v_new = B^T * u_vec
            Vector new_v(n, zero);
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < m; ++i) {
                    new_v[j] += b[i][j] * u_vec[i];
                }
            }

            Scalar diff = zero;
            for (std::size_t j = 0; j < n; ++j) {
                diff += (new_v[j] - v_vec[j]) * (new_v[j] - v_vec[j]);
            }

            v_vec = new_v;
            Scalar norm = std::sqrt(DotProduct(v_vec, v_vec));
            if (norm < kEpsilon) break;
            for (auto& val : v_vec) val /= norm;

            if (std::sqrt(diff) < kEpsilon) break;
        }

        // Compute singular value.
        Vector u_vec(m, zero);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                u_vec[i] += b[i][j] * v_vec[j];
            }
        }

        Scalar sigma = std::sqrt(DotProduct(u_vec, u_vec));
        if (sigma < kEpsilon) break;
        for (auto& val : u_vec) val /= sigma;

        s[idx] = sigma;
        for (std::size_t i = 0; i < m; ++i) u[i][idx] = u_vec[i];
        for (std::size_t j = 0; j < n; ++j) v[j][idx] = v_vec[j];

        // Deflate.
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                b[i][j] -= sigma * u_vec[i] * v_vec[j];
            }
        }
    }

    // Sort descending by singular value.
    std::vector<std::size_t> order(k);
    for (std::size_t i = 0; i < k; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                  return s[lhs] > s[rhs];
              });

    Vector sorted_s(k);
    Matrix sorted_u(m, Vector(m, zero));
    Matrix sorted_v(n, Vector(n, zero));
    // Copy non-SVD columns as-is from identity.
    for (std::size_t i = 0; i < m; ++i) sorted_u[i][i] = one;
    for (std::size_t i = 0; i < n; ++i) sorted_v[i][i] = one;

    for (std::size_t i = 0; i < k; ++i) {
        sorted_s[i] = s[order[i]];
        for (std::size_t j = 0; j < m; ++j) {
            sorted_u[j][i] = u[j][order[i]];
        }
        for (std::size_t j = 0; j < n; ++j) {
            sorted_v[j][i] = v[j][order[i]];
        }
    }

    return {sorted_u, sorted_s, Transpose(sorted_v)};
}

// ----------------------------------------------------------------------------
// Derived quantities
// ----------------------------------------------------------------------------

template <std::floating_point Scalar>
typename SingularValueDecomposition<Scalar>::Matrix
SingularValueDecomposition<Scalar>::Reconstruct(const SvdResult& result) {
    std::size_t m = result.U.size();
    std::size_t n = result.Vt.empty() ? 0 : result.Vt[0].size();
    constexpr Scalar zero = static_cast<Scalar>(0);

    Matrix sigma(m, Vector(n, zero));
    for (std::size_t i = 0; i < result.SingularValues.size(); ++i) {
        sigma[i][i] = result.SingularValues[i];
    }
    auto u_sigma = MatrixMultiply(result.U, sigma);
    return MatrixMultiply(u_sigma, result.Vt);
}

template <std::floating_point Scalar>
typename SingularValueDecomposition<Scalar>::Matrix
SingularValueDecomposition<Scalar>::Pseudoinverse(const SvdResult& result,
                                                   Scalar tolerance) {
    std::size_t m = result.U.size();
    std::size_t n = result.Vt.empty() ? 0 : result.Vt[0].size();
    constexpr Scalar zero = static_cast<Scalar>(0);
    constexpr Scalar one  = static_cast<Scalar>(1);

    if (tolerance < zero) {
        tolerance = static_cast<Scalar>(std::max(m, n)) *
                    result.SingularValues[0] *
                    std::numeric_limits<Scalar>::epsilon();
    }

    // Vt is k x n, V is n x k.  We need V * SigmaInv * U^T.
    auto v = Transpose(result.Vt);

    Matrix sigma_inv(result.SingularValues.size(),
                     Vector(m, zero));
    for (std::size_t i = 0; i < result.SingularValues.size(); ++i) {
        if (result.SingularValues[i] > tolerance) {
            sigma_inv[i][i] = one / result.SingularValues[i];
        }
    }

    // Need n x k * k x m * m x m  =>  n x m
    // But sigma_inv is k x m sized.  We need to pad for the multiply.
    // Construct full sigma_inv as n x m.
    Matrix full_sigma_inv(n, Vector(m, zero));
    for (std::size_t i = 0; i < result.SingularValues.size(); ++i) {
        if (result.SingularValues[i] > tolerance) {
            full_sigma_inv[i][i] = one / result.SingularValues[i];
        }
    }

    auto v_sigma_inv = MatrixMultiply(v, full_sigma_inv);
    return MatrixMultiply(v_sigma_inv, Transpose(result.U));
}

template <std::floating_point Scalar>
std::size_t SingularValueDecomposition<Scalar>::Rank(const SvdResult& result,
                                                      Scalar tolerance) {
    if (result.SingularValues.empty()) return 0;
    if (tolerance < static_cast<Scalar>(0)) {
        std::size_t max_dim = std::max(
            result.U.size(),
            result.Vt.empty() ? std::size_t{0} : result.Vt[0].size());
        tolerance = static_cast<Scalar>(max_dim) *
                    result.SingularValues[0] *
                    std::numeric_limits<Scalar>::epsilon();
    }
    std::size_t r = 0;
    for (Scalar sv : result.SingularValues) {
        if (sv > tolerance) ++r;
    }
    return r;
}

template <std::floating_point Scalar>
Scalar SingularValueDecomposition<Scalar>::ConditionNumber(
    const SvdResult& result) {
    if (result.SingularValues.empty()) {
        return std::numeric_limits<Scalar>::infinity();
    }
    Scalar smallest = std::numeric_limits<Scalar>::max();
    for (Scalar sv : result.SingularValues) {
        if (sv > kEpsilon && sv < smallest) smallest = sv;
    }
    if (smallest == std::numeric_limits<Scalar>::max()) {
        return std::numeric_limits<Scalar>::infinity();
    }
    return result.SingularValues[0] / smallest;
}

template <std::floating_point Scalar>
Scalar SingularValueDecomposition<Scalar>::FrobeniusNorm(
    const SvdResult& result) {
    Scalar sum = static_cast<Scalar>(0);
    for (Scalar sv : result.SingularValues) sum += sv * sv;
    return std::sqrt(sum);
}

#endif // SINGULAR_VALUE_DECOMPOSITION_HPP
