#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// Structure for a data point with two features and a binary label
struct DataPoint {
    double x1;  // Feature 1
    double x2;  // Feature 2
    int label;  // Binary label (0 or 1)
};

// Structure to hold class statistics
struct ClassStats {
    double mean_x1, mean_x2;  // Means of features
    int count;                // Number of samples in class
};

// Compute the mean of a vector
double compute_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

// LDA class for binary classification
class LDA {
private:
    double w1, w2;  // Weights for the projection vector
    double threshold;  // Classification threshold

public:
    // Train the LDA model on the dataset
    void fit(const std::vector<DataPoint>& data) {
        // Split data by class
        std::vector<DataPoint> class_0, class_1;
        for (const auto& dp : data) {
            if (dp.label == 0) class_0.push_back(dp);
            else class_1.push_back(dp);
        }

        // Compute class means
        ClassStats stats_0, stats_1;
        stats_0.count = class_0.size();
        stats_1.count = class_1.size();

        std::vector<double> x1_0, x2_0, x1_1, x2_1;
        for (const auto& dp : class_0) {
            x1_0.push_back(dp.x1);
            x2_0.push_back(dp.x2);
        }
        for (const auto& dp : class_1) {
            x1_1.push_back(dp.x1);
            x2_1.push_back(dp.x2);
        }

        stats_0.mean_x1 = compute_mean(x1_0);
        stats_0.mean_x2 = compute_mean(x2_0);
        stats_1.mean_x1 = compute_mean(x1_1);
        stats_1.mean_x2 = compute_mean(x2_1);

        // Compute within-class scatter (simplified variance)
        double sw_x1 = 0.0, sw_x2 = 0.0;
        for (const auto& dp : class_0) {
            sw_x1 += std::pow(dp.x1 - stats_0.mean_x1, 2);
            sw_x2 += std::pow(dp.x2 - stats_0.mean_x2, 2);
        }
        for (const auto& dp : class_1) {
            sw_x1 += std::pow(dp.x1 - stats_1.mean_x1, 2);
            sw_x2 += std::pow(dp.x2 - stats_1.mean_x2, 2);
        }
        sw_x1 /= (data.size() - 2);  // Degrees of freedom
        sw_x2 /= (data.size() - 2);

        // Compute projection vector w = (mean1 - mean0) / scatter
        double diff_x1 = stats_1.mean_x1 - stats_0.mean_x1;
        double diff_x2 = stats_1.mean_x2 - stats_0.mean_x2;
        w1 = diff_x1 / sw_x1;
        w2 = diff_x2 / sw_x2;

        // Normalize the projection vector
        double norm = std::sqrt(w1 * w1 + w2 * w2);
        w1 /= norm;
        w2 /= norm;

        // Compute projections of class means and set threshold
        double proj_mean_0 = w1 * stats_0.mean_x1 + w2 * stats_0.mean_x2;
        double proj_mean_1 = w1 * stats_1.mean_x1 + w2 * stats_1.mean_x2;
        threshold = (proj_mean_0 + proj_mean_1) / 2.0;
    }

    // Project a data point onto the LDA axis
    double project(const DataPoint& dp) const {
        return w1 * dp.x1 + w2 * dp.x2;
    }

    // Predict the class of a new data point
    int predict(const DataPoint& dp) const {
        double projection = project(dp);
        return (projection >= threshold) ? 1 : 0;
    }

    // Display learned parameters
    void print_parameters() const {
        std::cout << "Projection vector: w1 = " << w1 << ", w2 = " << w2 << "\n";
        std::cout << "Classification threshold: " << threshold << "\n";
    }
};

int main() {
    // Hardcoded sample dataset: {x1, x2, label}
    std::vector<DataPoint> dataset = {
        {2.0, 3.0, 0},
        {1.0, 2.0, 0},
        {3.0, 4.0, 0},
        {5.0, 6.0, 1},
        {4.0, 5.0, 1},
        {6.0, 7.0, 1}
    };

    // Train the LDA model
    LDA model;
    model.fit(dataset);

    // Show the learned parameters
    model.print_parameters();

    // Get user input for prediction
    std::cout << "Enter x1 and x2 (e.g., 3.5 4.5): ";
    double x1, x2;
    std::cin >> x1 >> x2;
    DataPoint new_point = {x1, x2, -1};  // Label is unused

    // Predict and display the result
    int prediction = model.predict(new_point);
    std::cout << "Predicted class: " << prediction << "\n";

    return 0;
}