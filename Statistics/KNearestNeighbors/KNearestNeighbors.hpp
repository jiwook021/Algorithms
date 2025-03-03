/**
 * @file KNearestNeighbors.hpp
 * @brief N-dimensional K-Nearest Neighbors (KNN) classifier.
 *
 * @details
 * KNN is a non-parametric, instance-based learning algorithm. Classification
 * is performed by majority vote among the k nearest training examples
 * in the feature space, using Euclidean distance.
 *
 * Decision rule:
 *   y_hat = argmax_c sum_{i in N_k(x)} I(y_i == c)
 *
 * where N_k(x) is the set of k nearest neighbors of x.
 *
 * This implementation generalizes to N-dimensional feature vectors and uses
 * std::partial_sort to efficiently find the K nearest neighbors without
 * fully sorting the entire training set.
 */

#ifndef K_NEAREST_NEIGHBORS_HPP
#define K_NEAREST_NEIGHBORS_HPP

#include <concepts>
#include <cstddef>
#include <vector>

/**
 * @class KNearestNeighbors
 * @brief Implements the k-NN classifier for N-dimensional, multi-class data.
 *
 * @tparam Scalar Floating-point type for feature values (default: double).
 */
template <std::floating_point Scalar = double>
class KNearestNeighbors {
public:
    using Vector = std::vector<Scalar>;

    /**
     * @brief A data point with N-dimensional features and a class label.
     */
    struct DataPoint {
        Vector features;  ///< N-dimensional feature vector
        int label;        ///< Class label (integer)
    };

    /**
     * @brief Construct a KNN classifier with a given k.
     * @param k Number of neighbors to consider.
     */
    explicit KNearestNeighbors(std::size_t k);

    /**
     * @brief Store the training data for later prediction.
     * @param training_data Vector of labeled data points.
     */
    void Fit(std::vector<DataPoint> training_data);

    /**
     * @brief Predict the class label for a single query point.
     * @param query N-dimensional feature vector.
     * @return Predicted class label (majority vote among k nearest).
     */
    int Predict(const Vector& query) const;

    /**
     * @brief Predict class labels for multiple query points.
     * @param queries Vector of N-dimensional feature vectors.
     * @return Vector of predicted class labels.
     */
    std::vector<int> PredictBatch(const std::vector<Vector>& queries) const;

    /**
     * @brief Compute Euclidean distance between two N-dimensional vectors.
     * @param a First vector.
     * @param b Second vector.
     * @return Euclidean distance ||a - b||_2.
     */
    static Scalar EuclideanDistance(const Vector& a, const Vector& b);

private:
    std::size_t k_;                        ///< Number of neighbors
    std::vector<DataPoint> training_data_; ///< Stored training examples
};

#endif // K_NEAREST_NEIGHBORS_HPP
