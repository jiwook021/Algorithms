#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// Struct for a data point with two features and a binary label
struct DataPoint {
    double x1;  // Feature 1
    double x2;  // Feature 2
    int label;  // Binary label (0 or 1)
};

// Struct to store statistics for each class
struct ClassStats {
    double mean_x1, mean_x2;  // Means of features
    double var_x1, var_x2;    // Variances of features
    double prior;             // Prior probability of the class
};

// Gaussian probability density function
double gaussian_prob(double x, double mean, double variance) {
    const double PI = 3.141592653589793;
    double std_dev = std::sqrt(variance);
    return (1.0 / (std_dev * std::sqrt(2.0 * PI))) * 
           std::exp(-std::pow(x - mean, 2) / (2.0 * variance));
}

// Compute the mean of a vector
double compute_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

// Compute the variance of a vector given its mean
double compute_variance(const std::vector<double>& values, double mean) {
    if (values.size() <= 1) return 0.0;
    double sum_sq_diff = 0.0;
    for (double v : values) {
        sum_sq_diff += std::pow(v - mean, 2);
    }
    return sum_sq_diff / (values.size() - 1);  // Sample variance
}

// Train the classifier
std::pair<ClassStats, ClassStats> train(const std::vector<DataPoint>& data) {
    std::vector<DataPoint> class_0, class_1;

    // Split data by class
    for (const auto& dp : data) {
        if (dp.label == 0) class_0.push_back(dp);
        else class_1.push_back(dp);
    }

    // Compute priors
    double prior_0 = static_cast<double>(class_0.size()) / data.size();
    double prior_1 = static_cast<double>(class_1.size()) / data.size();

    // Class 0 statistics
    std::vector<double> x1_0, x2_0;
    for (const auto& dp : class_0) {
        x1_0.push_back(dp.x1);
        x2_0.push_back(dp.x2);
    }
    double mean_x1_0 = compute_mean(x1_0);
    double mean_x2_0 = compute_mean(x2_0);
    double var_x1_0 = compute_variance(x1_0, mean_x1_0);
    double var_x2_0 = compute_variance(x2_0, mean_x2_0);

    // Class 1 statistics
    std::vector<double> x1_1, x2_1;
    for (const auto& dp : class_1) {
        x1_1.push_back(dp.x1);
        x2_1.push_back(dp.x2);
    }
    double mean_x1_1 = compute_mean(x1_1);
    double mean_x2_1 = compute_mean(x2_1);
    double var_x1_1 = compute_variance(x1_1, mean_x1_1);
    double var_x2_1 = compute_variance(x2_1, mean_x2_1);

    // Prevent zero variance
    const double MIN_VARIANCE = 1e-4;
    var_x1_0 = std::max(var_x1_0, MIN_VARIANCE);
    var_x2_0 = std::max(var_x2_0, MIN_VARIANCE);
    var_x1_1 = std::max(var_x1_1, MIN_VARIANCE);
    var_x2_1 = std::max(var_x2_1, MIN_VARIANCE);

    return {
        {mean_x1_0, mean_x2_0, var_x1_0, var_x2_0, prior_0},  // Class 0
        {mean_x1_1, mean_x2_1, var_x1_1, var_x2_1, prior_1}   // Class 1
    };
}

// Predict the class for a new data point
int predict(const ClassStats& stats_0, const ClassStats& stats_1, const DataPoint& dp) {
    // Score for class 0
    double prob_x1_0 = gaussian_prob(dp.x1, stats_0.mean_x1, stats_0.var_x1);
    double prob_x2_0 = gaussian_prob(dp.x2, stats_0.mean_x2, stats_0.var_x2);
    double score_0 = std::log(stats_0.prior) + std::log(prob_x1_0) + std::log(prob_x2_0);

    // Score for class 1
    double prob_x1_1 = gaussian_prob(dp.x1, stats_1.mean_x1, stats_1.var_x1);
    double prob_x2_1 = gaussian_prob(dp.x2, stats_1.mean_x2, stats_1.var_x2);
    double score_1 = std::log(stats_1.prior) + std::log(prob_x1_1) + std::log(prob_x2_1);

    // Pick the higher score
    return (score_1 > score_0) ? 1 : 0;
}

int main() {
    // Sample dataset: {x1, x2, label}
    std::vector<DataPoint> dataset = {
        {2.0, 3.0, 0},
        {1.0, 2.0, 0},
        {3.0, 4.0, 0},
        {5.0, 6.0, 1},
        {4.0, 5.0, 1},
        {6.0, 7.0, 1}
    };

    // Train the model
    auto [stats_0, stats_1] = train(dataset);

    // Get user input
    std::cout << "Enter x1 and x2 (e.g., 3.5 4.5): ";
    double x1, x2;
    std::cin >> x1 >> x2;
    DataPoint new_dp = {x1, x2, -1};  // Label is ignored

    // Predict and show result
    int prediction = predict(stats_0, stats_1, new_dp);
    std::cout << "Predicted class: " << prediction << std::endl;

    return 0;
}