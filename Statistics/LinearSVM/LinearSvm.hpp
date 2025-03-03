/**
 * @file LinearSvm.hpp
 * @brief N-dimensional Linear Support Vector Machine with sub-gradient descent.
 *
 * @details
 * SVM finds the maximum-margin hyperplane separating two classes.
 * The primal optimization problem (soft-margin):
 *   min_{w,b} (1/2)||w||^2 + C * sum max(0, 1 - y_i(w^T x_i + b))
 *
 * This implementation uses sub-gradient descent on the hinge loss.
 * Templatized on Scalar type (must satisfy std::floating_point).
 */

#ifndef LINEAR_SVM_HPP
#define LINEAR_SVM_HPP

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * @class LinearSvm
 * @brief N-dimensional linear SVM classifier trained via sub-gradient descent.
 *
 * @tparam Scalar Floating-point type for weights and computations.
 *
 * Train minimizes:  (1/2)||w||^2 + C * sum_i max(0, 1 - y_i (w . x_i + b))
 * using sub-gradient descent.  Each epoch iterates over every data point
 * and applies the appropriate sub-gradient update.
 */
template <std::floating_point Scalar = double>
class LinearSvm {
public:
    using Vector = std::vector<Scalar>;

    /** @brief A labeled data point: feature vector + class label in {-1, +1}. */
    struct DataPoint {
        Vector features;
        int label;  ///< Must be +1 or -1
    };

    /** @brief Default constructor. */
    LinearSvm() = default;

    /**
     * @brief Train the SVM on labeled data using sub-gradient descent.
     *
     * @param data            Training data with labels in {-1, +1}.
     * @param learning_rate   Step size for gradient descent.
     * @param epochs          Number of full passes over the dataset.
     * @param regularization_C Regularization parameter C (higher = less regularization,
     *                         i.e., more emphasis on fitting training data).
     *
     * @throws std::invalid_argument if data is empty or dimensions are inconsistent.
     */
    void Train(const std::vector<DataPoint>& data,
               Scalar learning_rate,
               int epochs,
               Scalar regularization_C);

    /**
     * @brief Predict the class label for a feature vector.
     * @return +1 or -1.
     */
    int Predict(const Vector& features) const;

    /**
     * @brief Compute the raw decision function value: w . x + b.
     */
    Scalar DecisionFunction(const Vector& features) const;

    /** @brief Get the learned weight vector. */
    const Vector& GetWeights() const;

    /** @brief Get the learned bias term. */
    Scalar GetBias() const;

private:
    Vector weights_;
    Scalar bias_{0};
};

// ---------------------------------------------------------------------------
// Template implementation (must be in header)
// ---------------------------------------------------------------------------

template <std::floating_point Scalar>
void LinearSvm<Scalar>::Train(const std::vector<DataPoint>& data,
                              Scalar learning_rate,
                              int epochs,
                              Scalar regularization_C) {
    if (data.empty()) {
        throw std::invalid_argument("Training data must not be empty.");
    }

    const std::size_t dim = data[0].features.size();
    for (const auto& dp : data) {
        if (dp.features.size() != dim) {
            throw std::invalid_argument(
                "All data points must have the same feature dimension.");
        }
    }

    // Initialize weights to zero
    weights_.assign(dim, Scalar{0});
    bias_ = Scalar{0};

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& dp : data) {
            Scalar y = static_cast<Scalar>(dp.label);

            // Compute w . x + b
            Scalar score = bias_;
            for (std::size_t j = 0; j < dim; ++j) {
                score += weights_[j] * dp.features[j];
            }

            if (y * score < Scalar{1}) {
                // Within margin or misclassified: hinge-loss sub-gradient
                for (std::size_t j = 0; j < dim; ++j) {
                    weights_[j] += learning_rate *
                        (y * dp.features[j] -
                         (Scalar{1} / regularization_C) * weights_[j]);
                }
                bias_ += learning_rate * y;
            } else {
                // Correctly classified outside margin: regularization only
                for (std::size_t j = 0; j < dim; ++j) {
                    weights_[j] -= learning_rate *
                        (Scalar{1} / regularization_C) * weights_[j];
                }
            }
        }
    }
}

template <std::floating_point Scalar>
int LinearSvm<Scalar>::Predict(const Vector& features) const {
    return DecisionFunction(features) >= Scalar{0} ? 1 : -1;
}

template <std::floating_point Scalar>
Scalar LinearSvm<Scalar>::DecisionFunction(const Vector& features) const {
    Scalar score = bias_;
    for (std::size_t j = 0; j < weights_.size(); ++j) {
        score += weights_[j] * features[j];
    }
    return score;
}

template <std::floating_point Scalar>
const typename LinearSvm<Scalar>::Vector& LinearSvm<Scalar>::GetWeights() const {
    return weights_;
}

template <std::floating_point Scalar>
Scalar LinearSvm<Scalar>::GetBias() const {
    return bias_;
}

#endif // LINEAR_SVM_HPP
