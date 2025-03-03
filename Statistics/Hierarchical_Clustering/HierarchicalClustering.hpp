/**
 * @file HierarchicalClustering.hpp
 * @brief Agglomerative hierarchical clustering with single-linkage.
 *
 * @details
 * Hierarchical clustering builds a dendrogram by iteratively merging the
 * two closest clusters until a single cluster remains (or a target number
 * of clusters is reached).
 *
 * Single-linkage distance: d(C1, C2) = min_{p1 in C1, p2 in C2} ||p1 - p2||
 */

#ifndef HIERARCHICAL_CLUSTERING_HPP
#define HIERARCHICAL_CLUSTERING_HPP

#include <limits>
#include <utility>
#include <vector>

/**
 * @brief Represents a 2D point.
 */
struct Point {
    double X;  ///< X-coordinate
    double Y;  ///< Y-coordinate
};

/**
 * @brief Represents a cluster of 2D points.
 */
struct Cluster {
    std::vector<Point> Points;  ///< Points in this cluster
};

/**
 * @class HierarchicalClustering
 * @brief Performs agglomerative hierarchical clustering with single-linkage.
 */
class HierarchicalClustering {
public:
    /**
     * @brief Compute Euclidean distance between two points.
     */
    static double Distance(const Point& p1, const Point& p2);

    /**
     * @brief Compute single-linkage distance between two clusters.
     */
    static double ClusterDistance(const Cluster& c1, const Cluster& c2);

    /**
     * @brief Find the pair of closest clusters.
     * @param clusters Vector of current clusters.
     * @return Pair of indices (i, j) of the closest clusters.
     */
    static std::pair<int, int> FindClosestClusters(
        const std::vector<Cluster>& clusters);

    /**
     * @brief Merge two clusters into one.
     */
    static Cluster MergeClusters(const Cluster& c1, const Cluster& c2);

    /**
     * @brief Run hierarchical clustering until targetClusters remain.
     * @param data Input points.
     * @param targetClusters Number of clusters to stop at (default: 1).
     * @return Final clusters.
     */
    static std::vector<Cluster> Fit(const std::vector<Point>& data,
                                     int targetClusters = 1);
};

#endif // HIERARCHICAL_CLUSTERING_HPP
