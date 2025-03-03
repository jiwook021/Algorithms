/**
 * @file NaiveBayesClassifier.hpp
 * @brief Gaussian Naive Bayes classifier for N features, K classes.
 *
 * @details
 * Assumes features are conditionally independent given the class and
 * follow a Gaussian distribution. Uses log-space Bayes' theorem:
 *   log P(C_k | x) = log P(C_k) + sum_j log N(x_j | mu_{kj}, sigma_{kj}^2)
 *
 * A variance floor prevents numerical issues (division by zero / log of zero).
 */

#ifndef NAIVE_BAYES_CLASSIFIER_HPP
#define NAIVE_BAYES_CLASSIFIER_HPP

#include <concepts>
#include <unordered_map>
#include <vector>

/**
 * @class NaiveBayesClassifier
 * @brief Gaussian Naive Bayes supporting N features and K classes with
 *        log-probability numerics.
 * @tparam Scalar Floating-point type used for all calculations.
 */
template <std::floating_point Scalar = double>
class NaiveBayesClassifier {
public:
    using Vector = std::vector<Scalar>;

    /**
     * @brief A data point: feature vector + integer class label.
     */
    struct DataPoint {
        Vector features;
        int label;
    };

    /**
     * @brief Train the classifier on labeled data.
     *
     * Computes per-class mean, variance (with floor), and log-prior
     * for every feature dimension.
     *
     * @param data Training dataset.
     */
    void Fit(const std::vector<DataPoint>& data);

    /**
     * @brief Predict the most likely class for a feature vector.
     * @param features Input feature vector.
     * @return Predicted class label.
     */
    int Predict(const Vector& features) const;

    /**
     * @brief Compute unnormalized log-posterior for every known class.
     *
     * log P(class | x) = log P(class) + sum_j log N(x_j | mu_j, var_j)
     *
     * @param features Input feature vector.
     * @return Map-ordered vector of (class_label, log_posterior) pairs
     *         stored as a simple vector of Scalar values keyed the same
     *         way as class_stats_.
     */
    std::vector<Scalar> PredictLogProbabilities(const Vector& features) const;

    /**
     * @brief Read-only access to learned class statistics (for testing).
     */
    const auto& GetClassStats() const { return class_stats_; }

private:
    /**
     * @brief Per-class sufficient statistics learned during Fit().
     */
    struct ClassStatistics {
        Vector mean;
        Vector variance;
        Scalar log_prior;
    };

    /**
     * @brief Log of the Gaussian PDF evaluated at x.
     *
     * log N(x | mu, var) = -0.5 * [ log(2*pi*var) + (x - mu)^2 / var ]
     */
    static Scalar LogGaussianPdf(Scalar x, Scalar mean, Scalar variance);

    /// Minimum allowed variance to prevent division by zero / log(0).
    static constexpr Scalar kVarianceFloor = static_cast<Scalar>(1e-9);

    /// Learned statistics indexed by class label.
    std::unordered_map<int, ClassStatistics> class_stats_;
};

#endif // NAIVE_BAYES_CLASSIFIER_HPP
