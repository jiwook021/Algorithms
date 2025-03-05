#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// Structure for a 2D point
struct Point {
    double x, y;
    Point(double x, double y) : x(x), y(y) {}
};

// Structure for a cluster (list of points)
struct Cluster {
    std::vector<Point> points;
    Cluster(const Point& p) { points.push_back(p); }
};

// Euclidean distance between two points
double distance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

// Minimum distance between two clusters (single-linkage)
double cluster_distance(const Cluster& c1, const Cluster& c2) {
    double min_dist = std::numeric_limits<double>::max();
    for (const auto& p1 : c1.points) {
        for (const auto& p2 : c2.points) {
            double dist = distance(p1, p2);
            if (dist < min_dist) min_dist = dist;
        }
    }
    return min_dist;
}

// Find the pair of clusters with the smallest distance
std::pair<int, int> find_closest_clusters(const std::vector<Cluster>& clusters) {
    double min_dist = std::numeric_limits<double>::max();
    std::pair<int, int> closest_pair(-1, -1);
    for (int i = 0; i < clusters.size(); ++i) {
        for (int j = i + 1; j < clusters.size(); ++j) {
            double dist = cluster_distance(clusters[i], clusters[j]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_pair = {i, j};
            }
        }
    }
    return closest_pair;
}

// Merge two clusters into one
Cluster merge_clusters(const Cluster& c1, const Cluster& c2) {
    Cluster new_cluster = c1;
    new_cluster.points.insert(new_cluster.points.end(), c2.points.begin(), c2.points.end());
    return new_cluster;
}

// Hierarchical clustering algorithm
void hierarchical_clustering(const std::vector<Point>& data) {
    // Initialize each point as its own cluster
    std::vector<Cluster> clusters;
    for (const auto& p : data) {
        clusters.push_back(Cluster(p));
    }

    while (clusters.size() > 1) {
        // Find the closest pair of clusters
        auto [i, j] = find_closest_clusters(clusters);
        if (i == -1 || j == -1) break;  // No more pairs to merge

        // Merge the closest clusters
        Cluster merged = merge_clusters(clusters[i], clusters[j]);

        // Remove the merged clusters and add the new one
        clusters.erase(clusters.begin() + std::max(i, j));
        clusters.erase(clusters.begin() + std::min(i, j));
        clusters.push_back(merged);

        // Output the merge step
        std::cout << "Merged clusters with points:\n";
        for (const auto& p : merged.points) {
            std::cout << "  (" << p.x << ", " << p.y << ")\n";
        }
        std::cout << "Remaining clusters: " << clusters.size() << "\n\n";
    }
}

int main() {
    // Hardcoded dataset: 2D points
    std::vector<Point> dataset = {
        {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0}, {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}
    };

    std::cout << "Starting hierarchical clustering...\n\n";
    hierarchical_clustering(dataset);

    return 0;
}