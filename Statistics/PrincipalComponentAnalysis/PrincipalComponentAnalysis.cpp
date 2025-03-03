/**
 * @file PrincipalComponentAnalysis.cpp
 * @brief Implementation of N-dimensional PCA via Jacobi eigendecomposition.
 *
 * @details
 * Steps:
 *   1. Center the data (subtract per-feature mean).
 *   2. Compute the N_features x N_features covariance matrix.
 *   3. Eigendecompose the covariance via Jacobi rotations.
 *   4. Sort eigenpairs by descending eigenvalue.
 *   5. Store components (eigenvectors as rows) and explained variance ratios.
 *
 * The Jacobi method iteratively applies 2D Givens rotations to drive
 * off-diagonal elements toward zero. It converges quadratically for
 * all real symmetric matrices.
 */

#include "PrincipalComponentAnalysis.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// -----------------------------------------------------------------------
// Public interface
// -----------------------------------------------------------------------

template <std::floating_point Scalar>
void PrincipalComponentAnalysis<Scalar>::Fit(const Matrix& data) {
    if (data.empty() || data[0].empty()) {
        throw std::invalid_argument("PCA::Fit: data must be non-empty");
    }

    const std::size_t n_features = data[0].size();

    // 1. Center the data
    Matrix centered = data;
    mean_ = CenterData(centered);

    // 2. Compute covariance matrix
    Matrix cov = ComputeCovarianceMatrix(centered);

    // 3. Eigendecompose via Jacobi
    Vector eigenvalues;
    Matrix eigenvectors;
    JacobiEigendecomposition(cov, eigenvalues, eigenvectors);

    // 4. Sort eigenpairs by descending eigenvalue
    std::vector<std::size_t> indices(n_features);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&eigenvalues](std::size_t a, std::size_t b) {
                  return eigenvalues[a] > eigenvalues[b];
              });

    // 5. Store components as rows (each row is an eigenvector)
    components_.resize(n_features, Vector(n_features));
    Vector sorted_eigenvalues(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        sorted_eigenvalues[i] = eigenvalues[indices[i]];
        for (std::size_t j = 0; j < n_features; ++j) {
            // eigenvectors are stored as columns
            components_[i][j] = eigenvectors[j][indices[i]];
        }
    }

    // 6. Compute explained variance ratios
    Scalar total_variance = static_cast<Scalar>(0);
    for (auto val : sorted_eigenvalues) {
        total_variance += std::max(val, static_cast<Scalar>(0));
    }
    explained_variance_.resize(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        if (total_variance > static_cast<Scalar>(0)) {
            explained_variance_[i] =
                std::max(sorted_eigenvalues[i], static_cast<Scalar>(0)) /
                total_variance;
        } else {
            explained_variance_[i] = static_cast<Scalar>(0);
        }
    }

    fitted_ = true;
}

template <std::floating_point Scalar>
typename PrincipalComponentAnalysis<Scalar>::Matrix
PrincipalComponentAnalysis<Scalar>::Transform(const Matrix& data,
                                              std::size_t n_components) const {
    if (!fitted_) {
        throw std::runtime_error("PCA::Transform: must call Fit first");
    }
    if (n_components > components_.size()) {
        throw std::invalid_argument(
            "PCA::Transform: n_components exceeds number of features");
    }

    const std::size_t n_samples = data.size();
    const std::size_t n_features = data[0].size();

    // Project: (data - mean) * components^T
    Matrix result(n_samples, Vector(n_components, static_cast<Scalar>(0)));
    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t k = 0; k < n_components; ++k) {
            Scalar dot = static_cast<Scalar>(0);
            for (std::size_t j = 0; j < n_features; ++j) {
                dot += (data[i][j] - mean_[j]) * components_[k][j];
            }
            result[i][k] = dot;
        }
    }
    return result;
}

template <std::floating_point Scalar>
typename PrincipalComponentAnalysis<Scalar>::Matrix
PrincipalComponentAnalysis<Scalar>::FitTransform(const Matrix& data,
                                                 std::size_t n_components) {
    Fit(data);
    return Transform(data, n_components);
}

template <std::floating_point Scalar>
const typename PrincipalComponentAnalysis<Scalar>::Vector&
PrincipalComponentAnalysis<Scalar>::GetExplainedVarianceRatio() const {
    return explained_variance_;
}

template <std::floating_point Scalar>
const typename PrincipalComponentAnalysis<Scalar>::Matrix&
PrincipalComponentAnalysis<Scalar>::GetComponents() const {
    return components_;
}

template <std::floating_point Scalar>
const typename PrincipalComponentAnalysis<Scalar>::Vector&
PrincipalComponentAnalysis<Scalar>::GetMean() const {
    return mean_;
}

// -----------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------

template <std::floating_point Scalar>
typename PrincipalComponentAnalysis<Scalar>::Vector
PrincipalComponentAnalysis<Scalar>::CenterData(Matrix& data) const {
    const std::size_t n_samples = data.size();
    const std::size_t n_features = data[0].size();

    Vector mean(n_features, static_cast<Scalar>(0));
    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < n_features; ++j) {
            mean[j] += data[i][j];
        }
    }
    Scalar n = static_cast<Scalar>(n_samples);
    for (std::size_t j = 0; j < n_features; ++j) {
        mean[j] /= n;
    }

    // Subtract mean
    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < n_features; ++j) {
            data[i][j] -= mean[j];
        }
    }
    return mean;
}

template <std::floating_point Scalar>
typename PrincipalComponentAnalysis<Scalar>::Matrix
PrincipalComponentAnalysis<Scalar>::ComputeCovarianceMatrix(
    const Matrix& centered) const {
    const std::size_t n_samples = centered.size();
    const std::size_t n_features = centered[0].size();

    // Unbiased covariance: divide by (n - 1)
    Scalar divisor = static_cast<Scalar>(n_samples - 1);
    if (n_samples == 1) {
        divisor = static_cast<Scalar>(1);  // Avoid division by zero
    }

    Matrix cov(n_features, Vector(n_features, static_cast<Scalar>(0)));
    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t p = 0; p < n_features; ++p) {
            for (std::size_t q = p; q < n_features; ++q) {
                cov[p][q] += centered[i][p] * centered[i][q];
            }
        }
    }
    for (std::size_t p = 0; p < n_features; ++p) {
        for (std::size_t q = p; q < n_features; ++q) {
            cov[p][q] /= divisor;
            cov[q][p] = cov[p][q];  // Symmetric
        }
    }
    return cov;
}

template <std::floating_point Scalar>
void PrincipalComponentAnalysis<Scalar>::JacobiEigendecomposition(
    Matrix& matrix, Vector& eigenvalues, Matrix& eigenvectors) const {
    const std::size_t n = matrix.size();
    const int kMaxIterations = 200;
    const Scalar kTolerance = static_cast<Scalar>(1e-12);

    // Initialize eigenvectors to identity
    eigenvectors.assign(n, Vector(n, static_cast<Scalar>(0)));
    for (std::size_t i = 0; i < n; ++i) {
        eigenvectors[i][i] = static_cast<Scalar>(1);
    }

    for (int iter = 0; iter < kMaxIterations; ++iter) {
        // Find the largest off-diagonal element
        Scalar max_off_diag = static_cast<Scalar>(0);
        std::size_t p = 0, q = 1;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                Scalar val = std::abs(matrix[i][j]);
                if (val > max_off_diag) {
                    max_off_diag = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Check convergence
        if (max_off_diag < kTolerance) {
            break;
        }

        // Compute Jacobi rotation angle
        Scalar app = matrix[p][p];
        Scalar aqq = matrix[q][q];
        Scalar apq = matrix[p][q];

        Scalar theta;
        if (std::abs(app - aqq) < kTolerance) {
            theta = static_cast<Scalar>(M_PI / 4.0);
        } else {
            theta = static_cast<Scalar>(0.5) *
                    std::atan2(static_cast<Scalar>(2) * apq, app - aqq);
        }

        Scalar cos_t = std::cos(theta);
        Scalar sin_t = std::sin(theta);

        // Apply Givens rotation to matrix: G^T * A * G
        // Update rows/cols p and q
        Vector row_p(n), row_q(n);
        for (std::size_t i = 0; i < n; ++i) {
            row_p[i] = cos_t * matrix[p][i] + sin_t * matrix[q][i];
            row_q[i] = -sin_t * matrix[p][i] + cos_t * matrix[q][i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            matrix[p][i] = row_p[i];
            matrix[q][i] = row_q[i];
            matrix[i][p] = row_p[i];
            matrix[i][q] = row_q[i];
        }
        // Fix the 2x2 block
        matrix[p][p] = cos_t * row_p[p] + sin_t * row_p[q];
        matrix[q][q] = -sin_t * row_q[p] + cos_t * row_q[q];
        matrix[p][q] = static_cast<Scalar>(0);
        matrix[q][p] = static_cast<Scalar>(0);

        // Update eigenvectors: V = V * G
        for (std::size_t i = 0; i < n; ++i) {
            Scalar vip = eigenvectors[i][p];
            Scalar viq = eigenvectors[i][q];
            eigenvectors[i][p] = cos_t * vip + sin_t * viq;
            eigenvectors[i][q] = -sin_t * vip + cos_t * viq;
        }
    }

    // Extract eigenvalues from diagonal
    eigenvalues.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = matrix[i][i];
    }
}

// -----------------------------------------------------------------------
// Explicit template instantiations
// -----------------------------------------------------------------------
template class PrincipalComponentAnalysis<float>;
template class PrincipalComponentAnalysis<double>;
template class PrincipalComponentAnalysis<long double>;
