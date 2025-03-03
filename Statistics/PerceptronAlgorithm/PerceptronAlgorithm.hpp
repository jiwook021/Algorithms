/**
 * @file PerceptronAlgorithm.hpp
 * @brief Perceptron algorithm for binary classification.
 *
 * @details
 * The perceptron is a linear classifier that learns a separating hyperplane
 * through iterative weight updates. For each misclassified point:
 *   w <- w + lr * (y - y_hat) * x
 *   b <- b + lr * (y - y_hat)
 *
 * Converges in finite steps for linearly separable data (Perceptron
 * Convergence Theorem).
 */

#ifndef PERCEPTRON_ALGORITHM_HPP
#define PERCEPTRON_ALGORITHM_HPP

#include <vector>

/**
 * @brief A 2D data point with a binary label.
 */
struct DataPoint {
    double X1;   ///< Feature 1
    double X2;   ///< Feature 2
    int Label;   ///< Class label (0 or 1)
};

/**
 * @class PerceptronAlgorithm
 * @brief Single-layer perceptron for binary classification.
 */
class PerceptronAlgorithm {
public:
    /**
     * @brief Construct a perceptron.
     * @param learningRate Step size for weight updates.
     * @param epochs Number of training passes over the data.
     * @param w1Init Initial weight for feature 1.
     * @param w2Init Initial weight for feature 2.
     * @param biasInit Initial bias.
     */
    PerceptronAlgorithm(double learningRate = 0.1, int epochs = 100,
                         double w1Init = 0.1, double w2Init = 0.1,
                         double biasInit = 0.0);

    /**
     * @brief Step activation function: returns 1 if sum >= 0, else 0.
     */
    int StepFunction(double sum) const;

    /**
     * @brief Train the perceptron on labeled data.
     */
    void Train(const std::vector<DataPoint>& dataset);

    /**
     * @brief Predict the class (0 or 1) for a query point.
     */
    int Predict(double x1, double x2) const;

    /** @brief Get weight w1. */
    double GetW1() const;

    /** @brief Get weight w2. */
    double GetW2() const;

    /** @brief Get bias. */
    double GetBias() const;

private:
    double w1_;
    double w2_;
    double bias_;
    double learningRate_;
    int epochs_;
};

#endif // PERCEPTRON_ALGORITHM_HPP
