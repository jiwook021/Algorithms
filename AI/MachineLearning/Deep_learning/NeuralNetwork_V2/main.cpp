/**
 * @file main.cpp
 * @brief Driver for the NeuralNetworkV2 module.
 */
#include "NeuralNetworkV2.hpp"
#include <iostream>
#include <iomanip>
#include <random>

int main() {
    using namespace Ml;

    // Matrix demo
    std::cout << "=== Matrix Operations ===\n";
    Matrix a({{1, 2}, {3, 4}});
    Matrix b({{5, 6}, {7, 8}});
    std::cout << "A + B:\n" << (a + b);
    std::cout << "A * B:\n" << (a * b);

    // Linear Regression demo
    std::cout << "\n=== Linear Regression ===\n";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 0.5);

    std::vector<Vector> xTrain;
    std::vector<double> yTrain;
    for (int i = 0; i < 100; ++i) {
        Vector s(2);
        s.At(0) = i / 50.0;
        s.At(1) = i / 25.0;
        xTrain.push_back(s);
        yTrain.push_back(2 * s.At(0) + 3 * s.At(1) + noise(gen));
    }

    LinearRegression lr(2, 0.1, 1000);
    lr.Fit(xTrain, yTrain, true);
    std::cout << "Learned: w1=" << lr.GetWeights().At(0)
              << " w2=" << lr.GetWeights().At(1)
              << " b=" << lr.GetBias() << "\n";

    return 0;
}
