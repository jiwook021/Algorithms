/**
 * @file LogisticRegression.hpp
 * @brief N-dimensional Logistic Regression for binary classification.
 *
 * @details
 * Logistic regression models the probability of the positive class as:
 *   P(y=1 | x) = sigma(w^T x + b)
 *
 * where sigma(z) = 1 / (1 + exp(-z)) is the sigmoid function.
 * Training minimizes the binary cross-entropy loss via gradient descent
 * with optional early stopping when the loss improvement falls below
 * a configurable tolerance.
 *
 * Generalized to N-dimensional feature vectors using a concepts-constrained
 * template on the scalar type.
 */

#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <concepts>
#include <vector>

/**
 * @class LogisticRegression
 * @brief N-dimensional binary logistic regression classifier trained via gradient descent.
 * @tparam Scalar Floating-point type for weights and computations.
 */
template<std::floating_point Scalar = double>
class LogisticRegression {
public:
    using Vector = std::vector<Scalar>;

    /**
     * @brief A data point with an N-dimensional feature vector and binary label.
     */
    struct DataPoint {
        Vector features;  ///< N-dimensional feature vector
        int label;         ///< Class label (0 or 1)
    };

    /**
     * @brief Train the model on labeled data via gradient descent.
     * @param data          Training dataset (all feature vectors must have same dimensionality).
     * @param learning_rate Step size for gradient descent.
     * @param epochs        Maximum number of training passes over the data.
     * @param tolerance     Stop early when loss improvement is below this threshold.
     */
    void Fit(const std::vector<DataPoint>& data,
             Scalar learning_rate,
             int epochs,
             Scalar tolerance = static_cast<Scalar>(1e-6));

    /**
     * @brief Predict the probability of class 1.
     * @param features N-dimensional feature vector.
     * @return Probability in [0, 1].
     */
    Scalar PredictProbability(const Vector& features) const;

    /**
     * @brief Predict the class label (0 or 1).
     * @param features  N-dimensional feature vector.
     * @param threshold Decision boundary (default 0.5).
     * @return 0 or 1.
     */
    int PredictClass(const Vector& features, Scalar threshold = 0.5) const;

    /**
     * @brief Compute classification accuracy on a dataset.
     * @param data Labeled dataset.
     * @return Accuracy as a fraction in [0, 1].
     */
    Scalar CalculateAccuracy(const std::vector<DataPoint>& data) const;

    /**
     * @brief Compute binary cross-entropy loss on a dataset.
     * @param data Labeled dataset.
     * @return Average cross-entropy loss.
     */
    Scalar CrossEntropyLoss(const std::vector<DataPoint>& data) const;

    /** @brief Get the learned weight vector. */
    const Vector& GetWeights() const;

    /** @brief Get the learned bias term. */
    Scalar GetBias() const;

    /**
     * @brief Compute the sigmoid function.
     * @param z Input value.
     * @return Sigmoid(z) in (0, 1).
     */
    static Scalar Sigmoid(Scalar z);

private:
    Vector weights_;       ///< Weight vector (one per feature dimension)
    Scalar bias_{0};       ///< Bias (intercept) term
};

#endif // LOGISTIC_REGRESSION_HPP
