/**
 * @file GaussianMixtureModel.cpp
 * @brief Implementation of the K-component N-dimensional GMM with EM.
 */

#include "GaussianMixtureModel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

static constexpr double kPi = 3.141592653589793;

// ---- Variance floor to prevent singularities ----
static constexpr double kVarianceFloor = 1e-6;

// =============================================================================
// Constructor
// =============================================================================

template <std::floating_point Scalar>
GaussianMixtureModel<Scalar>::GaussianMixtureModel(std::size_t n_components,
                                                     int max_iterations,
                                                     Scalar tolerance)
    : n_components_(n_components),
      max_iterations_(max_iterations),
      tolerance_(tolerance) {
    if (n_components_ == 0) {
        throw std::invalid_argument("n_components must be >= 1");
    }
}

// =============================================================================
// Gaussian PDF (diagonal covariance)
// =============================================================================

template <std::floating_point Scalar>
Scalar GaussianMixtureModel<Scalar>::GaussianPdf(const Vector& x,
                                                   const Vector& mean,
                                                   const Vector& variance) {
    std::size_t dims = x.size();
    Scalar log_pdf = static_cast<Scalar>(0);
    for (std::size_t d = 0; d < dims; ++d) {
        Scalar var = std::max(variance[d], static_cast<Scalar>(kVarianceFloor));
        Scalar diff = x[d] - mean[d];
        log_pdf -= static_cast<Scalar>(0.5) * std::log(static_cast<Scalar>(2.0 * kPi) * var);
        log_pdf -= (diff * diff) / (static_cast<Scalar>(2.0) * var);
    }
    return std::exp(log_pdf);
}

// =============================================================================
// E-Step
// =============================================================================

template <std::floating_point Scalar>
void GaussianMixtureModel<Scalar>::EStep(const Matrix& data,
                                          Matrix& responsibilities) const {
    std::size_t n_samples = data.size();
    responsibilities.resize(n_samples, Vector(n_components_, static_cast<Scalar>(0)));

    for (std::size_t i = 0; i < n_samples; ++i) {
        Scalar total = static_cast<Scalar>(0);
        for (std::size_t k = 0; k < n_components_; ++k) {
            Scalar weighted_pdf = components_[k].weight *
                                  GaussianPdf(data[i], components_[k].mean,
                                              components_[k].variance);
            responsibilities[i][k] = weighted_pdf;
            total += weighted_pdf;
        }
        // Normalize
        if (total > static_cast<Scalar>(0)) {
            for (std::size_t k = 0; k < n_components_; ++k) {
                responsibilities[i][k] /= total;
            }
        } else {
            // Uniform assignment when density is effectively zero everywhere
            for (std::size_t k = 0; k < n_components_; ++k) {
                responsibilities[i][k] =
                    static_cast<Scalar>(1.0) / static_cast<Scalar>(n_components_);
            }
        }
    }
}

// =============================================================================
// M-Step
// =============================================================================

template <std::floating_point Scalar>
void GaussianMixtureModel<Scalar>::MStep(const Matrix& data,
                                          const Matrix& responsibilities) {
    std::size_t n_samples = data.size();
    std::size_t dims = data[0].size();

    for (std::size_t k = 0; k < n_components_; ++k) {
        Scalar n_k = static_cast<Scalar>(0);
        for (std::size_t i = 0; i < n_samples; ++i) {
            n_k += responsibilities[i][k];
        }

        // Update mean
        components_[k].mean.assign(dims, static_cast<Scalar>(0));
        for (std::size_t i = 0; i < n_samples; ++i) {
            for (std::size_t d = 0; d < dims; ++d) {
                components_[k].mean[d] += responsibilities[i][k] * data[i][d];
            }
        }
        for (std::size_t d = 0; d < dims; ++d) {
            components_[k].mean[d] /= (n_k + static_cast<Scalar>(1e-10));
        }

        // Update variance (diagonal)
        components_[k].variance.assign(dims, static_cast<Scalar>(0));
        for (std::size_t i = 0; i < n_samples; ++i) {
            for (std::size_t d = 0; d < dims; ++d) {
                Scalar diff = data[i][d] - components_[k].mean[d];
                components_[k].variance[d] += responsibilities[i][k] * diff * diff;
            }
        }
        for (std::size_t d = 0; d < dims; ++d) {
            components_[k].variance[d] /= (n_k + static_cast<Scalar>(1e-10));
            components_[k].variance[d] = std::max(components_[k].variance[d],
                                                    static_cast<Scalar>(kVarianceFloor));
        }

        // Update weight
        components_[k].weight = n_k / static_cast<Scalar>(n_samples);
    }
}

// =============================================================================
// Log-Likelihood
// =============================================================================

template <std::floating_point Scalar>
Scalar GaussianMixtureModel<Scalar>::LogLikelihood(const Matrix& data) const {
    Scalar ll = static_cast<Scalar>(0);
    for (const auto& x : data) {
        Scalar sample_likelihood = static_cast<Scalar>(0);
        for (std::size_t k = 0; k < n_components_; ++k) {
            sample_likelihood += components_[k].weight *
                                 GaussianPdf(x, components_[k].mean,
                                             components_[k].variance);
        }
        ll += std::log(sample_likelihood +
                       static_cast<Scalar>(std::numeric_limits<Scalar>::min()));
    }
    return ll;
}

// =============================================================================
// Fit (EM algorithm)
// =============================================================================

template <std::floating_point Scalar>
void GaussianMixtureModel<Scalar>::Fit(const Matrix& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data must not be empty");
    }

    std::size_t n_samples = data.size();
    std::size_t dims = data[0].size();

    // ---- Initialize components via K-means++ inspired seeding ----
    components_.resize(n_components_);

    // Use deterministic seeding for reproducibility
    std::mt19937 rng(42);

    // Pick initial means from the data using K-means++ style
    std::vector<std::size_t> chosen_indices;
    std::uniform_int_distribution<std::size_t> uniform_idx(0, n_samples - 1);
    chosen_indices.push_back(uniform_idx(rng));

    for (std::size_t k = 1; k < n_components_; ++k) {
        // Compute distance to nearest chosen center
        std::vector<Scalar> distances(n_samples,
                                       std::numeric_limits<Scalar>::max());
        for (std::size_t i = 0; i < n_samples; ++i) {
            for (auto idx : chosen_indices) {
                Scalar dist = static_cast<Scalar>(0);
                for (std::size_t d = 0; d < dims; ++d) {
                    Scalar diff = data[i][d] - data[idx][d];
                    dist += diff * diff;
                }
                distances[i] = std::min(distances[i], dist);
            }
        }
        // Weighted random sampling proportional to distance squared
        Scalar total_dist = std::accumulate(distances.begin(), distances.end(),
                                             static_cast<Scalar>(0));
        std::uniform_real_distribution<Scalar> uniform_real(
            static_cast<Scalar>(0), total_dist);
        Scalar threshold = uniform_real(rng);
        Scalar cumulative = static_cast<Scalar>(0);
        std::size_t next_idx = 0;
        for (std::size_t i = 0; i < n_samples; ++i) {
            cumulative += distances[i];
            if (cumulative >= threshold) {
                next_idx = i;
                break;
            }
        }
        chosen_indices.push_back(next_idx);
    }

    // Compute global variance for initialization
    Vector global_mean(dims, static_cast<Scalar>(0));
    for (const auto& x : data) {
        for (std::size_t d = 0; d < dims; ++d) {
            global_mean[d] += x[d];
        }
    }
    for (std::size_t d = 0; d < dims; ++d) {
        global_mean[d] /= static_cast<Scalar>(n_samples);
    }
    Vector global_var(dims, static_cast<Scalar>(0));
    for (const auto& x : data) {
        for (std::size_t d = 0; d < dims; ++d) {
            Scalar diff = x[d] - global_mean[d];
            global_var[d] += diff * diff;
        }
    }
    for (std::size_t d = 0; d < dims; ++d) {
        global_var[d] /= static_cast<Scalar>(n_samples);
        global_var[d] = std::max(global_var[d],
                                  static_cast<Scalar>(kVarianceFloor));
    }

    for (std::size_t k = 0; k < n_components_; ++k) {
        components_[k].mean = data[chosen_indices[k]];
        components_[k].variance = global_var;
        components_[k].weight =
            static_cast<Scalar>(1.0) / static_cast<Scalar>(n_components_);
    }

    // ---- EM iterations with log-likelihood convergence monitoring ----
    Matrix responsibilities;
    Scalar prev_ll = -std::numeric_limits<Scalar>::max();

    for (int iter = 0; iter < max_iterations_; ++iter) {
        EStep(data, responsibilities);
        MStep(data, responsibilities);

        Scalar ll = LogLikelihood(data);
        if (std::abs(ll - prev_ll) < tolerance_) {
            break;
        }
        prev_ll = ll;
    }
}

// =============================================================================
// Predict (hard assignment)
// =============================================================================

template <std::floating_point Scalar>
std::vector<int> GaussianMixtureModel<Scalar>::Predict(const Matrix& data) const {
    Matrix resp = Responsibilities(data);
    std::vector<int> labels(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        auto it = std::max_element(resp[i].begin(), resp[i].end());
        labels[i] = static_cast<int>(std::distance(resp[i].begin(), it));
    }
    return labels;
}

// =============================================================================
// Responsibilities (public interface)
// =============================================================================

template <std::floating_point Scalar>
typename GaussianMixtureModel<Scalar>::Matrix
GaussianMixtureModel<Scalar>::Responsibilities(const Matrix& data) const {
    Matrix resp;
    EStep(data, resp);
    return resp;
}

// =============================================================================
// GetComponents
// =============================================================================

template <std::floating_point Scalar>
const std::vector<typename GaussianMixtureModel<Scalar>::Component>&
GaussianMixtureModel<Scalar>::GetComponents() const {
    return components_;
}

// =============================================================================
// Explicit template instantiations
// =============================================================================

template class GaussianMixtureModel<float>;
template class GaussianMixtureModel<double>;
