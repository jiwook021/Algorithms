/**
 * @file KMeansClustering.cpp
 * @brief Template implementation and explicit instantiations for KMeansClustering.
 */

#include "KMeansClustering.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

// ===========================================================================
// Template implementation
// ===========================================================================

template <std::floating_point Scalar>
KMeansClustering<Scalar>::KMeansClustering(std::size_t k, int max_iterations,
                                           Scalar tolerance, unsigned int seed)
    : k_(k),
      max_iterations_(max_iterations),
      tolerance_(tolerance),
      seed_(seed) {}

template <std::floating_point Scalar>
Scalar KMeansClustering<Scalar>::Distance(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "Distance: vectors must have the same dimensionality");
    }
    Scalar sum = static_cast<Scalar>(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        Scalar diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

template <std::floating_point Scalar>
typename KMeansClustering<Scalar>::Vector
KMeansClustering<Scalar>::ComputeCentroid(const Matrix& data,
                                          const std::vector<int>& assignments,
                                          int cluster) {
    if (data.empty()) {
        return {};
    }
    std::size_t dims = data[0].size();
    Vector centroid(dims, static_cast<Scalar>(0));
    int count = 0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        if (assignments[i] == cluster) {
            for (std::size_t d = 0; d < dims; ++d) {
                centroid[d] += data[i][d];
            }
            ++count;
        }
    }

    if (count == 0) {
        return centroid;  // zero vector -- empty cluster
    }

    for (std::size_t d = 0; d < dims; ++d) {
        centroid[d] /= static_cast<Scalar>(count);
    }
    return centroid;
}

template <std::floating_point Scalar>
std::vector<int> KMeansClustering<Scalar>::Fit(const Matrix& data) {
    if (data.empty()) {
        last_data_ = data;
        last_assignments_.clear();
        centroids_.clear();
        return {};
    }
    if (k_ == 0) {
        throw std::invalid_argument("Fit: k must be >= 1");
    }
    if (k_ > data.size()) {
        throw std::invalid_argument(
            "Fit: k must not exceed the number of data points");
    }

    last_data_ = data;
    std::size_t n = data.size();

    // Initialize centroids using k-means++ style random selection
    // (simple: pick k distinct random points)
    std::mt19937 rng(seed_);
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    centroids_.clear();
    centroids_.reserve(k_);
    for (std::size_t i = 0; i < k_; ++i) {
        centroids_.push_back(data[indices[i]]);
    }

    std::vector<int> assignments(n, -1);

    for (int iter = 0; iter < max_iterations_; ++iter) {
        // --- Assignment step ---
        for (std::size_t i = 0; i < n; ++i) {
            Scalar min_dist = std::numeric_limits<Scalar>::max();
            int best = 0;
            for (std::size_t c = 0; c < k_; ++c) {
                Scalar dist = Distance(data[i], centroids_[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best = static_cast<int>(c);
                }
            }
            assignments[i] = best;
        }

        // --- Update step ---
        Matrix new_centroids;
        new_centroids.reserve(k_);
        for (std::size_t c = 0; c < k_; ++c) {
            new_centroids.push_back(
                ComputeCentroid(data, assignments, static_cast<int>(c)));
        }

        // --- Convergence check: max centroid shift ---
        Scalar max_shift = static_cast<Scalar>(0);
        for (std::size_t c = 0; c < k_; ++c) {
            Scalar shift = Distance(centroids_[c], new_centroids[c]);
            if (shift > max_shift) {
                max_shift = shift;
            }
        }

        centroids_ = std::move(new_centroids);

        if (max_shift < tolerance_) {
            break;
        }
    }

    last_assignments_ = assignments;
    return assignments;
}

template <std::floating_point Scalar>
int KMeansClustering<Scalar>::Predict(const Vector& point) const {
    if (centroids_.empty()) {
        throw std::runtime_error("Predict: model has not been fitted yet");
    }
    Scalar min_dist = std::numeric_limits<Scalar>::max();
    int best = 0;
    for (std::size_t c = 0; c < centroids_.size(); ++c) {
        Scalar dist = Distance(point, centroids_[c]);
        if (dist < min_dist) {
            min_dist = dist;
            best = static_cast<int>(c);
        }
    }
    return best;
}

template <std::floating_point Scalar>
const typename KMeansClustering<Scalar>::Matrix&
KMeansClustering<Scalar>::GetCentroids() const {
    return centroids_;
}

template <std::floating_point Scalar>
Scalar KMeansClustering<Scalar>::Inertia() const {
    if (last_data_.empty()) {
        return static_cast<Scalar>(0);
    }
    Scalar inertia = static_cast<Scalar>(0);
    for (std::size_t i = 0; i < last_data_.size(); ++i) {
        int c = last_assignments_[i];
        Scalar dist = Distance(last_data_[i], centroids_[c]);
        inertia += dist * dist;
    }
    return inertia;
}

// ===========================================================================
// Explicit instantiation for common scalar types
// ===========================================================================

template class KMeansClustering<double>;
template class KMeansClustering<float>;
