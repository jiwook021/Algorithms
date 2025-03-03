/**
 * @file main.cpp
 * @brief Driver program for N-dimensional Logistic Regression.
 */

#include <iostream>
#include <vector>

#include "LogisticRegression.hpp"

int main() {
    using LR = LogisticRegression<double>;

    // 2D linearly-separable dataset
    std::vector<LR::DataPoint> dataset = {
        {{2.0, 3.0}, 0}, {{1.0, 2.0}, 0}, {{3.0, 4.0}, 0},
        {{5.0, 6.0}, 1}, {{4.0, 5.0}, 1}, {{6.0, 7.0}, 1}
    };

    LR model;
    model.Fit(dataset, 0.1, 1000);

    const auto& w = model.GetWeights();
    std::cout << "Weights:";
    for (std::size_t i = 0; i < w.size(); ++i) {
        std::cout << " w" << i << "=" << w[i];
    }
    std::cout << ", bias=" << model.GetBias() << "\n";

    std::cout << "Training accuracy: "
              << model.CalculateAccuracy(dataset) * 100.0 << "%\n";
    std::cout << "Cross-entropy loss: "
              << model.CrossEntropyLoss(dataset) << "\n";

    std::cout << "\nEnter two features (e.g., 3.5 4.5): ";
    double x1, x2;
    std::cin >> x1 >> x2;

    std::vector<double> query = {x1, x2};
    std::cout << "Probability of class 1: "
              << model.PredictProbability(query) << "\n";
    std::cout << "Predicted class: "
              << model.PredictClass(query) << "\n";

    return 0;
}
