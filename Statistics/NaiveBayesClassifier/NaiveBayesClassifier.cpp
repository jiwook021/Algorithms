/**
 * @file NaiveBayesClassifier.cpp
 * @brief Implementation of Gaussian Naive Bayes classifier (N features, K classes).
 */

#include "NaiveBayesClassifier.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <numbers>
#include <stdexcept>

// ---------------------------------------------------------------------------
// LogGaussianPdf
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
Scalar NaiveBayesClassifier<Scalar>::LogGaussianPdf(Scalar x, Scalar mean,
                                                     Scalar variance) {
    // log N(x | mu, var) = -0.5 * [ log(2*pi*var) + (x - mu)^2 / var ]
    const Scalar log_2pi_var = std::log(static_cast<Scalar>(2.0) *
                                        static_cast<Scalar>(std::numbers::pi) *
                                        variance);
    const Scalar diff = x - mean;
    return static_cast<Scalar>(-0.5) * (log_2pi_var + diff * diff / variance);
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
void NaiveBayesClassifier<Scalar>::Fit(const std::vector<DataPoint>& data) {
    if (data.empty()) {
        throw std::invalid_argument("NaiveBayesClassifier::Fit: empty dataset");
    }

    const std::size_t num_features = data.front().features.size();
    const auto total = static_cast<Scalar>(data.size());

    // Group data by label.
    std::unordered_map<int, std::vector<const DataPoint*>> groups;
    for (const auto& dp : data) {
        groups[dp.label].push_back(&dp);
    }

    class_stats_.clear();

    for (const auto& [label, points] : groups) {
        const auto count = static_cast<Scalar>(points.size());

        ClassStatistics stats;
        stats.mean.resize(num_features, static_cast<Scalar>(0));
        stats.variance.resize(num_features, static_cast<Scalar>(0));
        stats.log_prior = std::log(count / total);

        // Compute means.
        for (const auto* dp : points) {
            for (std::size_t j = 0; j < num_features; ++j) {
                stats.mean[j] += dp->features[j];
            }
        }
        for (std::size_t j = 0; j < num_features; ++j) {
            stats.mean[j] /= count;
        }

        // Compute sample variances (with floor).
        for (const auto* dp : points) {
            for (std::size_t j = 0; j < num_features; ++j) {
                const Scalar diff = dp->features[j] - stats.mean[j];
                stats.variance[j] += diff * diff;
            }
        }
        const Scalar denom = (count > static_cast<Scalar>(1))
                                 ? (count - static_cast<Scalar>(1))
                                 : static_cast<Scalar>(1);
        for (std::size_t j = 0; j < num_features; ++j) {
            stats.variance[j] = std::max(stats.variance[j] / denom,
                                         kVarianceFloor);
        }

        class_stats_[label] = std::move(stats);
    }
}

// ---------------------------------------------------------------------------
// PredictLogProbabilities
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
std::vector<Scalar>
NaiveBayesClassifier<Scalar>::PredictLogProbabilities(
    const Vector& features) const {
    if (class_stats_.empty()) {
        throw std::logic_error(
            "NaiveBayesClassifier::PredictLogProbabilities: model not fitted");
    }

    std::vector<Scalar> log_probs;
    log_probs.reserve(class_stats_.size());

    // Iterate in a deterministic order by collecting labels first.
    std::vector<int> labels;
    labels.reserve(class_stats_.size());
    for (const auto& [label, _] : class_stats_) {
        labels.push_back(label);
    }
    std::sort(labels.begin(), labels.end());

    for (int label : labels) {
        const auto& stats = class_stats_.at(label);
        Scalar score = stats.log_prior;
        for (std::size_t j = 0; j < features.size(); ++j) {
            score += LogGaussianPdf(features[j], stats.mean[j],
                                    stats.variance[j]);
        }
        log_probs.push_back(score);
    }

    return log_probs;
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------
template <std::floating_point Scalar>
int NaiveBayesClassifier<Scalar>::Predict(const Vector& features) const {
    if (class_stats_.empty()) {
        throw std::logic_error(
            "NaiveBayesClassifier::Predict: model not fitted");
    }

    int best_label = -1;
    Scalar best_score = -std::numeric_limits<Scalar>::infinity();

    for (const auto& [label, stats] : class_stats_) {
        Scalar score = stats.log_prior;
        for (std::size_t j = 0; j < features.size(); ++j) {
            score += LogGaussianPdf(features[j], stats.mean[j],
                                    stats.variance[j]);
        }
        if (score > best_score) {
            best_score = score;
            best_label = label;
        }
    }

    return best_label;
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------
template class NaiveBayesClassifier<double>;
template class NaiveBayesClassifier<float>;
