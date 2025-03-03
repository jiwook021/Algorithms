/**
 * @file HierarchicalClustering.cpp
 * @brief Implementation of agglomerative hierarchical clustering.
 */

#include "HierarchicalClustering.hpp"
#include <algorithm>
#include <cmath>

double HierarchicalClustering::Distance(const Point& p1, const Point& p2) {
    return std::sqrt((p1.X - p2.X) * (p1.X - p2.X) +
                     (p1.Y - p2.Y) * (p1.Y - p2.Y));
}

double HierarchicalClustering::ClusterDistance(const Cluster& c1,
                                                const Cluster& c2) {
    double minDist = std::numeric_limits<double>::max();
    for (const auto& p1 : c1.Points) {
        for (const auto& p2 : c2.Points) {
            double dist = Distance(p1, p2);
            if (dist < minDist) minDist = dist;
        }
    }
    return minDist;
}

std::pair<int, int> HierarchicalClustering::FindClosestClusters(
    const std::vector<Cluster>& clusters) {
    double minDist = std::numeric_limits<double>::max();
    int bestI = 0, bestJ = 1;
    for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(clusters.size()); ++j) {
            double dist = ClusterDistance(clusters[i], clusters[j]);
            if (dist < minDist) {
                minDist = dist;
                bestI = i;
                bestJ = j;
            }
        }
    }
    return {bestI, bestJ};
}

Cluster HierarchicalClustering::MergeClusters(const Cluster& c1,
                                               const Cluster& c2) {
    Cluster merged;
    merged.Points.insert(merged.Points.end(), c1.Points.begin(), c1.Points.end());
    merged.Points.insert(merged.Points.end(), c2.Points.begin(), c2.Points.end());
    return merged;
}

std::vector<Cluster> HierarchicalClustering::Fit(const std::vector<Point>& data,
                                                  int targetClusters) {
    std::vector<Cluster> clusters;
    for (const auto& p : data) {
        clusters.push_back({{p}});
    }

    while (static_cast<int>(clusters.size()) > targetClusters &&
           clusters.size() > 1) {
        auto [i, j] = FindClosestClusters(clusters);
        Cluster merged = MergeClusters(clusters[i], clusters[j]);
        clusters.erase(clusters.begin() + std::max(i, j));
        clusters.erase(clusters.begin() + std::min(i, j));
        clusters.push_back(merged);
    }

    return clusters;
}
