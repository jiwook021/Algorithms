/**
 * @file LinearDiscriminantAnalysis.cpp
 * @brief Implementation of N-dimensional Fisher's LDA.
 *
 * @details
 * Key steps:
 * 1. Compute class means mu_0, mu_1.
 * 2. Build the within-class scatter matrix S_w (full NxN).
 * 3. Solve w = S_w^{-1} (mu_1 - mu_0) via Gaussian elimination with
 *    partial pivoting.
 * 4. Normalize w to unit length.
 * 5. Set threshold at midpoint of projected class means.
 */

#include "LinearDiscriminantAnalysis.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// ---------------------------------------------------------------------------
// ComputeMean
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
auto LinearDiscriminantAnalysis<Scalar>::ComputeMean(
    const std::vector<DataPoint>& data, int label) const -> Vector {

    size_t dim = data.front().features.size();
    Vector mean(dim, Scalar{0});
    size_t count = 0;

    for (const auto& dp : data) {
        if (dp.label == label) {
            for (size_t i = 0; i < dim; ++i) {
                mean[i] += dp.features[i];
            }
            ++count;
        }
    }

    if (count == 0) {
        throw std::runtime_error("No data points found for label " +
                                 std::to_string(label));
    }

    for (size_t i = 0; i < dim; ++i) {
        mean[i] /= static_cast<Scalar>(count);
    }
    return mean;
}

// ---------------------------------------------------------------------------
// ComputeWithinClassScatter
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
auto LinearDiscriminantAnalysis<Scalar>::ComputeWithinClassScatter(
    const std::vector<DataPoint>& data) const -> Matrix {

    size_t dim = data.front().features.size();
    Matrix scatter(dim, Vector(dim, Scalar{0}));

    // Pre-compute means for both classes
    Vector mean0 = ComputeMean(data, 0);
    Vector mean1 = ComputeMean(data, 1);

    for (const auto& dp : data) {
        const Vector& mu = (dp.label == 0) ? mean0 : mean1;

        // Outer product accumulation: S_w += (x - mu)(x - mu)^T
        for (size_t i = 0; i < dim; ++i) {
            Scalar diff_i = dp.features[i] - mu[i];
            for (size_t j = 0; j < dim; ++j) {
                Scalar diff_j = dp.features[j] - mu[j];
                scatter[i][j] += diff_i * diff_j;
            }
        }
    }

    return scatter;
}

// ---------------------------------------------------------------------------
// SolveForProjection  --  Gaussian elimination with partial pivoting
// Solves: scatter * w = mean_diff   =>   w = scatter^{-1} * mean_diff
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
auto LinearDiscriminantAnalysis<Scalar>::SolveForProjection(
    const Matrix& scatter, const Vector& mean_diff) const -> Vector {

    size_t n = mean_diff.size();

    // Build augmented matrix [scatter | mean_diff]
    Matrix aug(n, Vector(n + 1, Scalar{0}));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = scatter[i][j];
        }
        aug[i][n] = mean_diff[i];
    }

    // Forward elimination with partial pivoting
    for (size_t col = 0; col < n; ++col) {
        // Find pivot
        size_t max_row = col;
        Scalar max_val = std::fabs(aug[col][col]);
        for (size_t row = col + 1; row < n; ++row) {
            Scalar v = std::fabs(aug[row][col]);
            if (v > max_val) {
                max_val = v;
                max_row = row;
            }
        }
        if (max_val < Scalar{1e-12}) {
            // Near-singular: add small regularization to diagonal
            aug[col][col] += Scalar{1e-8};
        }
        if (max_row != col) {
            std::swap(aug[col], aug[max_row]);
        }

        // Eliminate below
        for (size_t row = col + 1; row < n; ++row) {
            Scalar factor = aug[row][col] / aug[col][col];
            for (size_t j = col; j <= n; ++j) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    Vector w(n, Scalar{0});
    for (size_t i = n; i-- > 0;) {
        Scalar sum = aug[i][n];
        for (size_t j = i + 1; j < n; ++j) {
            sum -= aug[i][j] * w[j];
        }
        w[i] = sum / aug[i][i];
    }

    return w;
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
void LinearDiscriminantAnalysis<Scalar>::Fit(
    const std::vector<DataPoint>& data) {

    if (data.empty()) {
        throw std::runtime_error("Cannot fit LDA on empty dataset");
    }

    size_t dim = data.front().features.size();
    if (dim == 0) {
        throw std::runtime_error("Feature dimension must be at least 1");
    }

    // 1. Class means
    Vector mean0 = ComputeMean(data, 0);
    Vector mean1 = ComputeMean(data, 1);

    // 2. Mean difference: mu_1 - mu_0
    Vector mean_diff(dim);
    for (size_t i = 0; i < dim; ++i) {
        mean_diff[i] = mean1[i] - mean0[i];
    }

    // 3. Within-class scatter matrix
    Matrix scatter = ComputeWithinClassScatter(data);

    // 4. Solve for projection vector w = S_w^{-1} (mu_1 - mu_0)
    projection_vector_ = SolveForProjection(scatter, mean_diff);

    // 5. Normalize to unit length
    Scalar norm{0};
    for (size_t i = 0; i < dim; ++i) {
        norm += projection_vector_[i] * projection_vector_[i];
    }
    norm = std::sqrt(norm);
    if (norm > Scalar{1e-12}) {
        for (size_t i = 0; i < dim; ++i) {
            projection_vector_[i] /= norm;
        }
    }

    // 6. Threshold at midpoint of projected class means
    Scalar proj_mean0{0}, proj_mean1{0};
    for (size_t i = 0; i < dim; ++i) {
        proj_mean0 += projection_vector_[i] * mean0[i];
        proj_mean1 += projection_vector_[i] * mean1[i];
    }
    threshold_ = (proj_mean0 + proj_mean1) / Scalar{2};
}

// ---------------------------------------------------------------------------
// Project  --  returns a 1-element vector (the scalar projection)
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
auto LinearDiscriminantAnalysis<Scalar>::Project(
    const Vector& features) const -> Vector {

    Scalar val{0};
    for (size_t i = 0; i < features.size(); ++i) {
        val += projection_vector_[i] * features[i];
    }
    return {val};
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
int LinearDiscriminantAnalysis<Scalar>::Predict(
    const Vector& features) const {

    Scalar projected = Project(features)[0];
    return (projected >= threshold_) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
auto LinearDiscriminantAnalysis<Scalar>::GetProjectionVector() const
    -> const Vector& {
    return projection_vector_;
}

template <std::floating_point Scalar>
Scalar LinearDiscriminantAnalysis<Scalar>::GetThreshold() const {
    return threshold_;
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
template class LinearDiscriminantAnalysis<double>;
template class LinearDiscriminantAnalysis<float>;
