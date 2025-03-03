/**
 * @file KMeansClustering.hpp
 * @brief N-dimensional K-Means clustering with convergence tolerance.
 *
 * @details
 * K-Means groups data points into K clusters.  It works by repeatedly
 * assigning each point to the nearest cluster center (centroid), then
 * moving each centroid to the average of its assigned points.  This
 * continues until the centroids stop moving.
 *
 * The quality of a clustering is measured by inertia: the total
 * squared distance from every point to its assigned centroid.
 * Lower inertia means tighter, better-separated clusters.
 *
 * Templatized on any floating-point scalar (double, float, long double).
 */

#ifndef K_MEANS_CLUSTERING_HPP
#define K_MEANS_CLUSTERING_HPP

#include <concepts>
#include <cstddef>
#include <vector>

/**
 * @class KMeansClustering
 * @brief Performs K-Means clustering on N-dimensional data.
 *
 * @tparam Scalar Floating-point type for coordinates and distances.
 */
template <std::floating_point Scalar = double>
class KMeansClustering {
public:
    using Vector = std::vector<Scalar>;
    using Matrix = std::vector<Vector>;

    /**
     * @brief Construct a KMeansClustering model.
     * @param k             Number of clusters.
     * @param max_iterations Maximum iterations before stopping.
     * @param tolerance      Convergence tolerance on centroid shift.
     * @param seed           Random seed for centroid initialization.
     */
    KMeansClustering(std::size_t k, int max_iterations = 100,
                     Scalar tolerance = static_cast<Scalar>(1e-6),
                     unsigned int seed = 42);

    /**
     * @brief Fit the model to data (returns cluster assignments).
     * @param data  Matrix of data points (each row is a point).
     * @return Vector of cluster assignments (one per data point).
     */
    std::vector<int> Fit(const Matrix& data);

    /**
     * @brief Predict the cluster for a new point.
     * @param point The new data point.
     * @return Cluster index.
     */
    int Predict(const Vector& point) const;

    /**
     * @brief Get the current centroids (cluster centers).
     */
    const Matrix& GetCentroids() const;

    /**
     * @brief Compute the inertia (total squared distance from each
     *        point to its assigned centroid).  Lower = tighter clusters.
     *
     * Must be called after Fit(). Uses the last assignments and centroids.
     */
    Scalar Inertia() const;

    /**
     * @brief Compute the straight-line (Euclidean) distance between
     *        two N-dimensional points.
     */
    static Scalar Distance(const Vector& a, const Vector& b);

private:
    /**
     * @brief Compute the centroid (average position) of all points
     *        assigned to a given cluster.  Called during the update
     *        step to move each centroid to the center of its group.
     */
    Vector ComputeCentroid(const Matrix& data,
                           const std::vector<int>& assignments,
                           int cluster);

    std::size_t k_;
    int max_iterations_;
    Scalar tolerance_;
    unsigned int seed_;
    Matrix centroids_;
    Matrix last_data_;
    std::vector<int> last_assignments_;
};

#endif // K_MEANS_CLUSTERING_HPP
