/**
 * @file main.cpp
 * @brief Driver program for N-dimensional Linear Discriminant Analysis.
 *
 * Demonstrates Fisher's LDA on a small 2D dataset, then classifies a
 * user-supplied point.
 */

#include <iostream>
#include <vector>

#include "LinearDiscriminantAnalysis.hpp"

int main() {
    using LDA = LinearDiscriminantAnalysis<double>;
    using DP  = LDA::DataPoint;

    std::vector<DP> dataset = {
        {{2.0, 3.0}, 0}, {{1.0, 2.0}, 0}, {{3.0, 4.0}, 0},
        {{5.0, 6.0}, 1}, {{4.0, 5.0}, 1}, {{6.0, 7.0}, 1}
    };

    LDA model;
    model.Fit(dataset);

    const auto& w = model.GetProjectionVector();
    std::cout << "Projection vector: [";
    for (size_t i = 0; i < w.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << w[i];
    }
    std::cout << "]\n";
    std::cout << "Classification threshold: " << model.GetThreshold() << "\n";

    std::cout << "Enter x1 and x2 (e.g., 3.5 4.5): ";
    double x1, x2;
    std::cin >> x1 >> x2;

    std::cout << "Predicted class: "
              << model.Predict({x1, x2}) << "\n";

    return 0;
}
