/**
 * @file main.cpp
 * @brief Driver program for N-dimensional Linear SVM.
 */

#include <iostream>
#include <vector>

#include "LinearSvm.hpp"

int main() {
    using Svm = LinearSvm<double>;

    // 2D dataset (same spirit as the original demo)
    std::vector<Svm::DataPoint> dataset = {
        {{2.0, 3.0}, -1}, {{3.0, 3.0}, -1}, {{3.0, 4.0}, -1},
        {{5.0, 5.0},  1}, {{6.0, 5.0},  1}, {{7.0, 6.0},  1}
    };

    Svm model;
    model.Train(dataset, /*learning_rate=*/0.01, /*epochs=*/2000,
                /*regularization_C=*/1.0);

    const auto& w = model.GetWeights();
    std::cout << "Weights:";
    for (std::size_t i = 0; i < w.size(); ++i) {
        std::cout << " w" << i << "=" << w[i];
    }
    std::cout << ", bias=" << model.GetBias() << "\n";

    std::cout << "Enter " << w.size() << " feature values (space-separated): ";
    Svm::Vector query(w.size());
    for (auto& v : query) {
        std::cin >> v;
    }

    std::cout << "Decision value: " << model.DecisionFunction(query) << "\n";
    std::cout << "Predicted class: " << model.Predict(query) << "\n";

    return 0;
}
