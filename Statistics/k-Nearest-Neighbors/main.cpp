#include <iostream>
#include <vector>
#include <cmath>    // For sqrt() and pow()
#include <algorithm> // For sort()

// Structure for a data point with two features and a binary label
struct DataPoint {
    double x1;  // Feature 1
    double x2;  // Feature 2
    int label;  // Binary label (0 or 1)
};

// Structure to store distance and label for a neighbor
struct Neighbor {
    double distance;
    int label;
};

// k-Nearest Neighbors class
class KNN {
private:
    std::vector<DataPoint> dataset;
    int k;  // Number of neighbors to consider

    // Compute Euclidean distance between two points
    double euclidean_distance(double x1, double x2, double y1, double y2) const {
        return std::sqrt(std::pow(x1 - y1, 2) + std::pow(x2 - y2, 2));
    }

public:
    // Constructor: Initialize with dataset and k
    KNN(const std::vector<DataPoint>& data, int k_value) : dataset(data), k(k_value) {}

    // Predict the class for a new data point
    int predict(double x1, double x2) const {
        std::vector<Neighbor> neighbors;

        // Calculate distance to all points in the dataset
        for (const auto& dp : dataset) {
            double dist = euclidean_distance(x1, x2, dp.x1, dp.x2);
            neighbors.push_back({dist, dp.label});
        }

        // Sort neighbors by distance (ascending)
        std::sort(neighbors.begin(), neighbors.end(), 
                  [](const Neighbor& a, const Neighbor& b) {
                      return a.distance < b.distance;
                  });

        // Count votes among k nearest neighbors
        int count_0 = 0, count_1 = 0;
        for (int i = 0; i < k && i < neighbors.size(); i++) {
            if (neighbors[i].label == 0) {
                count_0++;
            } else {
                count_1++;
            }
        }

        // Return the majority class
        return (count_1 > count_0) ? 1 : 0;
    }
};

int main() {
    // Hardcoded dataset: {x1, x2, label}
    std::vector<DataPoint> dataset = {
        {2.0, 3.0, 0},
        {1.0, 1.0, 0},
        {3.0, 2.0, 0},
        {5.0, 4.0, 1},
        {6.0, 5.0, 1},
        {7.0, 3.0, 1}
    };

    // Initialize k-NN with k = 3
    KNN model(dataset, 3);

    // User input for prediction
    std::cout << "Enter x1 and x2 (e.g., 4.0 3.0): ";
    double x1, x2;
    std::cin >> x1 >> x2;

    // Predict and display the result
    int prediction = model.predict(x1, x2);
    std::cout << "Predicted class: " << prediction << std::endl;

    return 0;
}