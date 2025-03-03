/**
 * @file LogisticRegression.cpp
 * @brief Implementation of N-dimensional logistic regression via gradient descent.
 */

#include "LogisticRegression.hpp"

#include <cmath>
#include <stdexcept>

// ── Sigmoid ──────────────────────────────────────────────────────────────────

template<std::floating_point Scalar>
Scalar LogisticRegression<Scalar>::Sigmoid(Scalar z) {
    return static_cast<Scalar>(1) / (static_cast<Scalar>(1) + std::exp(-z));
}

// ── Fit ──────────────────────────────────────────────────────────────────────

template<std::floating_point Scalar>
void LogisticRegression<Scalar>::Fit(const std::vector<DataPoint>& data,
                                     Scalar learning_rate,
                                     int epochs,
                                     Scalar tolerance) {
    if (data.empty()) {
        throw std::invalid_argument("Training data must not be empty");
    }

    const std::size_t n_features = data[0].features.size();
    weights_.assign(n_features, static_cast<Scalar>(0));
    bias_ = static_cast<Scalar>(0);

    Scalar prev_loss = CrossEntropyLoss(data);

    for (int e = 0; e < epochs; ++e) {
        // Stochastic gradient descent (online, one sample at a time)
        for (const auto& dp : data) {
            Scalar pred = PredictProbability(dp.features);
            Scalar error = static_cast<Scalar>(dp.label) - pred;

            for (std::size_t j = 0; j < n_features; ++j) {
                weights_[j] += learning_rate * error * dp.features[j];
            }
            bias_ += learning_rate * error;
        }

        // Convergence check
        Scalar current_loss = CrossEntropyLoss(data);
        if (std::abs(prev_loss - current_loss) < tolerance) {
            break;
        }
        prev_loss = current_loss;
    }
}

// ── PredictProbability ───────────────────────────────────────────────────────

template<std::floating_point Scalar>
Scalar LogisticRegression<Scalar>::PredictProbability(const Vector& features) const {
    Scalar z = bias_;
    for (std::size_t j = 0; j < weights_.size(); ++j) {
        z += weights_[j] * features[j];
    }
    return Sigmoid(z);
}

// ── PredictClass ─────────────────────────────────────────────────────────────

template<std::floating_point Scalar>
int LogisticRegression<Scalar>::PredictClass(const Vector& features,
                                              Scalar threshold) const {
    return PredictProbability(features) >= threshold ? 1 : 0;
}

// ── CalculateAccuracy ────────────────────────────────────────────────────────

template<std::floating_point Scalar>
Scalar LogisticRegression<Scalar>::CalculateAccuracy(
    const std::vector<DataPoint>& data) const {
    int correct = 0;
    for (const auto& dp : data) {
        if (PredictClass(dp.features) == dp.label) {
            ++correct;
        }
    }
    return static_cast<Scalar>(correct) / static_cast<Scalar>(data.size());
}

// ── CrossEntropyLoss ─────────────────────────────────────────────────────────

template<std::floating_point Scalar>
Scalar LogisticRegression<Scalar>::CrossEntropyLoss(
    const std::vector<DataPoint>& data) const {
    constexpr Scalar kEpsilon = static_cast<Scalar>(1e-15);
    Scalar total = static_cast<Scalar>(0);

    for (const auto& dp : data) {
        Scalar p = PredictProbability(dp.features);
        // Clamp to avoid log(0)
        p = std::max(kEpsilon, std::min(static_cast<Scalar>(1) - kEpsilon, p));
        Scalar y = static_cast<Scalar>(dp.label);
        total -= y * std::log(p) + (static_cast<Scalar>(1) - y) * std::log(static_cast<Scalar>(1) - p);
    }
    return total / static_cast<Scalar>(data.size());
}

// ── Accessors ────────────────────────────────────────────────────────────────

template<std::floating_point Scalar>
const typename LogisticRegression<Scalar>::Vector&
LogisticRegression<Scalar>::GetWeights() const {
    return weights_;
}

template<std::floating_point Scalar>
Scalar LogisticRegression<Scalar>::GetBias() const {
    return bias_;
}

// ── Explicit instantiations ──────────────────────────────────────────────────

template class LogisticRegression<float>;
template class LogisticRegression<double>;
