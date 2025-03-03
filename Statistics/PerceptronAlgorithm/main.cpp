/**
 * @file main.cpp
 * @brief Driver program for the Perceptron Algorithm.
 */

#include <iostream>
#include <vector>

#include "PerceptronAlgorithm.hpp"

int main() {
    std::vector<DataPoint> dataset = {
        {2.0, 6.0, 0}, {4.0, 5.0, 0}, {3.0, 7.0, 0},
        {5.0, 4.0, 1}, {6.0, 6.0, 1}, {7.0, 5.0, 1}
    };

    PerceptronAlgorithm model(0.1, 100);
    model.Train(dataset);

    std::cout << "Learned weights: w1=" << model.GetW1()
              << ", w2=" << model.GetW2()
              << ", bias=" << model.GetBias() << std::endl;

    std::cout << "Enter study hours and sleep hours (e.g., 5.0 6.0): ";
    double x1, x2;
    std::cin >> x1 >> x2;

    int prediction = model.Predict(x1, x2);
    std::cout << "Predicted class (0=fail, 1=pass): " << prediction << std::endl;

    return 0;
}
