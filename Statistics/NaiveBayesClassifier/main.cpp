/**
 * @file main.cpp
 * @brief Driver program for Naive Bayes Classifier (N features, K classes).
 */

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NaiveBayesClassifier.hpp"

int main() {
    // Training data: 2D, 3 classes (well separated clusters).
    using NBC = NaiveBayesClassifier<double>;
    using DP  = NBC::DataPoint;

    std::vector<DP> dataset = {
        // Class 0 -- near origin
        {{1.0, 2.0}, 0}, {{2.0, 3.0}, 0}, {{1.5, 2.5}, 0},
        // Class 1 -- upper-right
        {{7.0, 8.0}, 1}, {{8.0, 9.0}, 1}, {{7.5, 8.5}, 1},
        // Class 2 -- lower-right
        {{8.0, 1.0}, 2}, {{9.0, 2.0}, 2}, {{8.5, 1.5}, 2},
    };

    NBC nbc;
    nbc.Fit(dataset);

    std::cout << "Gaussian Naive Bayes Classifier -- 3 classes, 2 features\n\n";
    std::cout << "Enter two features separated by space (e.g., 3.5 4.5): ";

    double x1, x2;
    if (!(std::cin >> x1 >> x2)) {
        std::cerr << "Invalid input.\n";
        return 1;
    }

    NBC::Vector query = {x1, x2};
    int prediction = nbc.Predict(query);

    auto log_probs = nbc.PredictLogProbabilities(query);
    std::cout << "Predicted class: " << prediction << "\n";
    std::cout << "Log-probabilities (sorted by class label):";
    for (auto lp : log_probs) {
        std::cout << " " << lp;
    }
    std::cout << "\n";

    return 0;
}
