/**
 * @file LinearDiscriminantAnalysis.hpp
 * @brief N-dimensional Fisher's Linear Discriminant Analysis for binary
 *        classification.
 *
 * @details
 * Fisher's LDA finds the projection vector w that maximizes class
 * separability in a lower-dimensional space:
 *
 *   w = S_w^{-1} (mu_1 - mu_0)
 *
 * where S_w is the within-class scatter matrix and mu_k are class means.
 * The within-class scatter matrix is:
 *
 *   S_w = sum_{k=0,1} sum_{x in C_k} (x - mu_k)(x - mu_k)^T
 *
 * Classification projects a sample onto w and compares to a threshold
 * set at the midpoint of the two projected class means.
 */

#ifndef LINEAR_DISCRIMINANT_ANALYSIS_HPP
#define LINEAR_DISCRIMINANT_ANALYSIS_HPP

#include <concepts>
#include <vector>

/**
 * @class LinearDiscriminantAnalysis
 * @brief N-dimensional binary LDA classifier using Fisher's criterion.
 * @tparam Scalar Floating-point type for all computations.
 */
template <std::floating_point Scalar = double>
class LinearDiscriminantAnalysis {
public:
    using Vector = std::vector<Scalar>;
    using Matrix = std::vector<Vector>;

    /** @brief A data point with an N-dimensional feature vector and label. */
    struct DataPoint {
        Vector features;  ///< Feature vector (N dimensions)
        int label;         ///< Class label (0 or 1)
    };

    /**
     * @brief Train the LDA model on labeled data.
     * @param data Training dataset (all feature vectors must have the same
     *             dimensionality and both classes must be present).
     */
    void Fit(const std::vector<DataPoint>& data);

    /**
     * @brief Project a feature vector onto the LDA discriminant axis.
     * @param features Feature vector (same dimensionality as training data).
     * @return Scalar projection value (dot product with w).
     */
    Vector Project(const Vector& features) const;

    /**
     * @brief Predict the class of a feature vector.
     * @param features Feature vector.
     * @return Predicted class (0 or 1).
     */
    int Predict(const Vector& features) const;

    /** @brief Get the projection (weight) vector. */
    const Vector& GetProjectionVector() const;

    /** @brief Get the classification threshold. */
    Scalar GetThreshold() const;

private:
    /**
     * @brief Compute the mean feature vector for a specific class.
     * @param data Full dataset.
     * @param label Class label to filter on.
     * @return Mean vector for that class.
     */
    Vector ComputeMean(const std::vector<DataPoint>& data, int label) const;

    /**
     * @brief Compute the within-class scatter matrix S_w.
     * @param data Full dataset.
     * @return S_w as a dense NxN matrix.
     */
    Matrix ComputeWithinClassScatter(const std::vector<DataPoint>& data) const;

    /**
     * @brief Solve w = S_w^{-1} * mean_diff via Gaussian elimination.
     * @param scatter The within-class scatter matrix (NxN).
     * @param mean_diff The difference vector mu_1 - mu_0.
     * @return The projection vector w.
     */
    Vector SolveForProjection(const Matrix& scatter,
                              const Vector& mean_diff) const;

    Vector projection_vector_;  ///< The learned projection direction
    Scalar threshold_{0};       ///< Decision boundary on the projected axis
};

#endif // LINEAR_DISCRIMINANT_ANALYSIS_HPP
