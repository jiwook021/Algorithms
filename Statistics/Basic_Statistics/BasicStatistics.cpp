/**
 * @file BasicStatistics.cpp
 * @brief Implementation of basic statistical functions.
 */

#include "BasicStatistics.hpp"

double BasicStatistics::Mean(const std::vector<double>& data) {
    double sum = 0.0;
    for (const double& val : data) {
        sum += val;
    }
    return sum / static_cast<double>(data.size());
}

double BasicStatistics::Slope(const std::vector<double>& x,
                               const std::vector<double>& y,
                               double meanX, double meanY) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - meanX;
        numerator += dx * (y[i] - meanY);
        denominator += dx * dx;
    }
    return numerator / denominator;
}

double BasicStatistics::Intercept(double meanY, double m, double meanX) {
    return meanY - m * meanX;
}

double BasicStatistics::Predict(double x, double m, double b) {
    return m * x + b;
}

double BasicStatistics::MeanSquaredError(const std::vector<double>& x,
                                          const std::vector<double>& y,
                                          double m, double b) {
    double sumError = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double error = y[i] - (m * x[i] + b);
        sumError += error * error;
    }
    return sumError / static_cast<double>(x.size());
}
