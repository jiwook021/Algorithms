/**
 * @file KNearestNeighbors.cpp
 * @brief Implementation of the N-dimensional K-Nearest Neighbors classifier.
 *
 * Uses std::partial_sort to efficiently select the K closest training
 * examples without sorting the entire distance list -- O(N log K) rather
 * than O(N log N).
 */

#include "KNearestNeighbors.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <std::floating_point Scalar>
KNearestNeighbors<Scalar>::KNearestNeighbors(std::size_t k) : k_(k) {
    if (k == 0) {
        throw std::invalid_argument("k must be >= 1");
    }
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

template <std::floating_point Scalar>
void KNearestNeighbors<Scalar>::Fit(std::vector<DataPoint> training_data) {
    if (training_data.empty()) {
        throw std::invalid_argument("Training data must not be empty");
    }
    training_data_ = std::move(training_data);
}

// ---------------------------------------------------------------------------
// Euclidean distance  ||a - b||_2
// ---------------------------------------------------------------------------

template <std::floating_point Scalar>
Scalar KNearestNeighbors<Scalar>::EuclideanDistance(const Vector& a,
                                                    const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Vectors must have the same dimensionality");
    }
    Scalar sum{};
    for (std::size_t i = 0; i < a.size(); ++i) {
        Scalar diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// ---------------------------------------------------------------------------
// Predict (single query) -- partial_sort for K nearest
// ---------------------------------------------------------------------------

template <std::floating_point Scalar>
int KNearestNeighbors<Scalar>::Predict(const Vector& query) const {
    if (training_data_.empty()) {
        throw std::logic_error("Must call Fit() before Predict()");
    }

    // Build (distance, label) pairs
    struct Neighbor {
        Scalar dist;
        int label;
    };

    std::vector<Neighbor> neighbors;
    neighbors.reserve(training_data_.size());
    for (const auto& dp : training_data_) {
        Scalar dist = EuclideanDistance(query, dp.features);
        neighbors.push_back({dist, dp.label});
    }

    // Use partial_sort: only the first k_ elements are in sorted order
    std::size_t k_actual = std::min(k_, neighbors.size());
    std::partial_sort(neighbors.begin(),
                      neighbors.begin() + static_cast<long>(k_actual),
                      neighbors.end(),
                      [](const Neighbor& a, const Neighbor& b) {
                          return a.dist < b.dist;
                      });

    // Majority vote among the k nearest
    std::unordered_map<int, int> votes;
    for (std::size_t i = 0; i < k_actual; ++i) {
        votes[neighbors[i].label]++;
    }

    int best_label = neighbors[0].label;
    int best_count = 0;
    for (const auto& [label, count] : votes) {
        if (count > best_count) {
            best_count = count;
            best_label = label;
        }
    }

    return best_label;
}

// ---------------------------------------------------------------------------
// PredictBatch
// ---------------------------------------------------------------------------

template <std::floating_point Scalar>
std::vector<int> KNearestNeighbors<Scalar>::PredictBatch(
    const std::vector<Vector>& queries) const {
    std::vector<int> results;
    results.reserve(queries.size());
    for (const auto& q : queries) {
        results.push_back(Predict(q));
    }
    return results;
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------
template class KNearestNeighbors<double>;
template class KNearestNeighbors<float>;
