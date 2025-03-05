#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Structure for a 2D point
struct Point {
    double x1, x2;  // Features
    int cluster;    // Assigned cluster (-1 means unassigned)
};

// Euclidean distance between two points
double distance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x1 - p2.x1, 2) + std::pow(p1.x2 - p2.x2, 2));
}

// Compute the mean point (centroid) for a cluster
Point compute_centroid(const std::vector<Point>& points, int cluster) {
    double sum_x1 = 0.0, sum_x2 = 0.0;
    int count = 0;
    for (const auto& p : points) {
        if (p.cluster == cluster) {
            sum_x1 += p.x1;
            sum_x2 += p.x2;
            count++;
        }
    }
    if (count == 0) return {0.0, 0.0, cluster};  // Avoid division by zero
    return {sum_x1 / count, sum_x2 / count, cluster};
}

// K-Means clustering algorithm
void k_means(std::vector<Point>& data, int k, int max_iterations = 100) {
    // Seed random number generator
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Initialize centroids randomly from data points
    std::vector<Point> centroids;
    for (int i = 0; i < k; ++i) {
        int idx = std::rand() % data.size();
        centroids.push_back({data[idx].x1, data[idx].x2, i});
    }

    bool changed = true;
    int iterations = 0;

    while (changed && iterations < max_iterations) {
        changed = false;
        iterations++;

        // Step 1: Assign points to nearest centroid
        for (auto& point : data) {
            double min_dist = distance(point, centroids[0]);
            int new_cluster = 0;

            for (int c = 1; c < k; ++c) {
                double dist = distance(point, centroids[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    new_cluster = c;
                }
            }

            if (point.cluster != new_cluster) {
                point.cluster = new_cluster;
                changed = true;
            }
        }

        // Step 2: Update centroids
        for (int c = 0; c < k; ++c) {
            centroids[c] = compute_centroid(data, c);
        }
    }

    std::cout << "Converged after " << iterations << " iterations.\n";
}

// Predict cluster for a new point
int predict_cluster(const Point& new_point, const std::vector<Point>& centroids) {
    double min_dist = distance(new_point, centroids[0]);
    int cluster = 0;

    for (int c = 1; c < static_cast<int>(centroids.size()); ++c) {
        double dist = distance(new_point, centroids[c]);
        if (dist < min_dist) {
            min_dist = dist;
            cluster = c;
        }
    }
    return cluster;
}

int main() {
    // Hardcoded dataset: {x1, x2, cluster (initially unassigned)}
    std::vector<Point> dataset = {
        {1.0, 2.0, -1},
        {1.5, 1.8, -1},
        {5.0, 8.0, -1},
        {8.0, 8.0, -1},
        {1.0, 0.6, -1},
        {9.0, 11.0, -1}
    };

    // Number of clusters
    int k = 2;

    // Run K-Means
    k_means(dataset, k);

    // Display the results
    std::cout << "Final clustering:\n";
    for (const auto& p : dataset) {
        std::cout << "Point (" << p.x1 << ", " << p.x2 << ") -> Cluster " << p.cluster << "\n";
    }

    // Store final centroids
    std::vector<Point> centroids(k);
    for (int c = 0; c < k; ++c) {
        centroids[c] = compute_centroid(dataset, c);
    }

    // User input for prediction
    std::cout << "\nEnter x1 and x2 to predict cluster (e.g., 3.0 4.0): ";
    double x1, x2;
    std::cin >> x1 >> x2;
    Point new_point = {x1, x2, -1};

    // Predict cluster for new point
    int pred_cluster = predict_cluster(new_point, centroids);
    std::cout << "Predicted cluster for (" << x1 << ", " << x2 << "): " << pred_cluster << "\n";

    return 0;
}