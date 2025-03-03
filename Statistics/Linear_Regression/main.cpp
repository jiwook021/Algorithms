/**
 * @file main.cpp
 * @brief Driver program for Linear Regression.
 */

#include <iostream>
#include <vector>

#include "LinearRegression.hpp"

int main() {
    std::vector<DataPoint> dataset = {
        {1.0, 2.1}, {2.0, 3.8}, {3.0, 5.2}, {4.0, 7.0}, {5.0, 8.9}
    };

    LinearRegression model;
    model.Fit(dataset);

    std::cout << "Learned model: y = " << model.GetSlope()
              << "x + " << model.GetIntercept() << std::endl;

    std::cout << "Enter an x value to predict y: ";
    double x;
    std::cin >> x;

    std::cout << "Predicted y for x = " << x << ": "
              << model.Predict(x) << std::endl;

    return 0;
}
