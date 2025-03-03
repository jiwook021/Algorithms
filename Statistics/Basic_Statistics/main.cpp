/**
 * @file main.cpp
 * @brief Driver program demonstrating basic statistics and linear regression.
 */

#include <iostream>
#include <vector>

#include "BasicStatistics.hpp"

int main() {
    std::vector<double> x = {1.0, 2.7, 3.0};
    std::vector<double> y = {2.0, 4.0, 6.0};

    double meanX = BasicStatistics::Mean(x);
    double meanY = BasicStatistics::Mean(y);

    double m = BasicStatistics::Slope(x, y, meanX, meanY);
    double b = BasicStatistics::Intercept(meanY, m, meanX);

    std::cout << "Slope (m): " << m << std::endl;
    std::cout << "Intercept (b): " << b << std::endl;

    double mse = BasicStatistics::MeanSquaredError(x, y, m, b);
    std::cout << "Mean Squared Error: " << mse << std::endl;

    std::cout << "Enter a new x value to predict y: ";
    double newX;
    std::cin >> newX;

    double predictedY = BasicStatistics::Predict(newX, m, b);
    std::cout << "Predicted y for x = " << newX << ": " << predictedY << std::endl;

    return 0;
}
