#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// Function to compute the mean of a vector
double compute_mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

// Function to compute the standard deviation of a vector
double compute_stddev(const std::vector<double>& data, double mean) {
    if (data.size() <= 1) return 0.0;
    double sum_sq_diff = 0.0;
    for (double val : data) {
        sum_sq_diff += std::pow(val - mean, 2);
    }
    return std::sqrt(sum_sq_diff / (data.size() - 1)); // Sample standard deviation
}

// Function to detect anomalies in the dataset
std::vector<bool> detect_anomalies(const std::vector<double>& data, double threshold) {
    double mean = compute_mean(data);
    double stddev = compute_stddev(data, mean);
    std::vector<bool> is_anomaly;
    for (double val : data) {
        bool anomaly = std::abs(val - mean) > threshold * stddev;
        is_anomaly.push_back(anomaly);
    }
    return is_anomaly;
}

// Function to check if a single value is an anomaly
bool is_anomaly(double value, double mean, double stddev, double threshold) {
    return std::abs(value - mean) > threshold * stddev;
}

int main() {
    // Hardcoded dataset with one obvious outlier
    std::vector<double> data = {10.0, 12.0, 11.0, 13.0, 9.0, 10.5, 11.5, 100.0, 10.2, 11.8};

    // Threshold: 2 standard deviations
    double threshold = 2.0;

    // Compute mean and standard deviation
    double mean = compute_mean(data);
    double stddev = compute_stddev(data, mean);

    // Detect anomalies
    std::vector<bool> anomalies = detect_anomalies(data, threshold);

    // Display dataset and results
    std::cout << "Dataset: ";
    for (double val : data) {
        std::cout << val << " ";
    }
    std::cout << "\n\nMean: " << mean << "\nStandard Deviation: " << stddev << "\n\n";

    std::cout << "Anomaly Detection Results:\n";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << "Value: " << data[i] << " -> " << (anomalies[i] ? "Anomaly" : "Normal") << "\n";
    }

    // User input to test a new value
    std::cout << "\nEnter a new value to check if it's an anomaly: ";
    double new_value;
    std::cin >> new_value;

    // Check and display result for the new value
    bool is_new_anomaly = is_anomaly(new_value, mean, stddev, threshold);
    std::cout << "The value " << new_value << " is " << (is_new_anomaly ? "an anomaly" : "normal") << ".\n";

    return 0;
}