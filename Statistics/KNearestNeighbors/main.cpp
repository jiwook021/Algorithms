/**
 * @file main.cpp
 * @brief Interactive driver for the K-Nearest Neighbors classifier.
 *
 * Loads a 2D training dataset with labeled classes, displays the data
 * grouped by class, and lets the user classify new points interactively.
 */

#include "KNearestNeighbors.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int main() {
    using KNN = KNearestNeighbors<double>;

    std::cout << "=============================================\n"
              << "  K-Nearest Neighbors Classifier\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  KNN is a non-parametric classifier — it has no training\n"
              << "  phase and makes no assumptions about the data distribution.\n"
              << "  To classify a new point, it finds the K closest training\n"
              << "  examples (using Euclidean distance) and assigns the class\n"
              << "  that appears most often among those K neighbors.\n\n"

              << "HOW IT WORKS\n"
              << "  1. Store all labeled training points.\n"
              << "  2. Given a query point, compute the distance to every\n"
              << "     training point.\n"
              << "  3. Pick the K nearest neighbors.\n"
              << "  4. The predicted class = majority vote among those K.\n\n"

              << "CHOOSING K\n"
              << "  Small K (e.g. 1) : sensitive to noise, sharp boundaries\n"
              << "  Large K (e.g. 7) : smoother boundaries, more robust\n"
              << "  Odd K avoids ties in binary classification.\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Training dataset — 2D points with 3 classes
    //
    //   Class 0 : bottom-left cluster   (around x=1, y=1)
    //   Class 1 : right cluster         (around x=8, y=3)
    //   Class 2 : top-left cluster      (around x=2, y=8)
    // -----------------------------------------------------------------

    std::vector<KNN::DataPoint> dataset = {
        // Class 0 — bottom-left
        {{1.0, 1.0}, 0},
        {{1.5, 2.0}, 0},
        {{2.0, 1.5}, 0},
        {{0.5, 0.5}, 0},

        // Class 1 — right
        {{7.0, 3.0}, 1},
        {{8.0, 2.5}, 1},
        {{8.5, 4.0}, 1},
        {{9.0, 3.5}, 1},

        // Class 2 — top-left
        {{1.5, 8.0}, 2},
        {{2.0, 9.0}, 2},
        {{2.5, 7.5}, 2},
        {{1.0, 8.5}, 2},
    };

    // Map class labels to human-readable names
    std::unordered_map<int, std::string> class_names = {
        {0, "Class 0 (bottom-left)"},
        {1, "Class 1 (right)"},
        {2, "Class 2 (top-left)"},
    };

    // Print the training data grouped by class
    std::cout << "Training Data  (" << dataset.size() << " points, 3 classes)\n\n";

    for (int c = 0; c <= 2; ++c) {
        std::cout << "  " << class_names[c] << ":\n";
        for (const auto& dp : dataset) {
            if (dp.label == c) {
                std::cout << "    ("
                          << std::fixed << std::setprecision(1)
                          << dp.features[0] << ", "
                          << dp.features[1] << ")\n";
            }
        }
    }

    // Visualize approximate layout
    std::cout << "\n  Approximate layout:\n"
              << "    y\n"
              << "    9 |       2 2\n"
              << "    8 |     2 2\n"
              << "    7 |\n"
              << "    6 |\n"
              << "    5 |\n"
              << "    4 |                   1\n"
              << "    3 |                 1   1\n"
              << "    2 |   0 0               1\n"
              << "    1 | 0 0\n"
              << "    0 +------------------------> x\n"
              << "      0  1  2  3  4  5  6  7  8  9\n\n";

    // Fit the model
    std::size_t k = 3;
    KNN model(k);
    model.Fit(dataset);

    std::cout << "Model: K = " << k
              << "  (majority vote among " << k << " nearest neighbors)\n\n";

    // Interactive query loop
    std::cout << "---------------------------------------------\n"
              << "  Enter a 2D point to classify, or 'q' to quit.\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  Query (x y): ";
        std::string input;
        std::getline(std::cin, input);

        if (input.empty() || input[0] == 'q' || input[0] == 'Q') {
            break;
        }

        double x, y;
        try {
            std::size_t pos1 = 0;
            x = std::stod(input, &pos1);
            y = std::stod(input.substr(pos1));
        } catch (...) {
            std::cout << "    Invalid input. Enter two numbers (e.g. 3.0 5.0).\n";
            continue;
        }

        int prediction = model.Predict({x, y});

        // Show distances to each class center for intuition
        std::cout << "    Point (" << std::fixed << std::setprecision(1)
                  << x << ", " << y << ")\n";

        // Find the K nearest neighbors for display
        struct Neighbor {
            double dist;
            int label;
            double fx, fy;
        };

        std::vector<Neighbor> neighbors;
        for (const auto& dp : dataset) {
            double dist = KNN::EuclideanDistance({x, y}, dp.features);
            neighbors.push_back({dist, dp.label, dp.features[0], dp.features[1]});
        }

        std::partial_sort(neighbors.begin(),
                          neighbors.begin() + static_cast<long>(k),
                          neighbors.end(),
                          [](const Neighbor& a, const Neighbor& b) {
                              return a.dist < b.dist;
                          });

        std::cout << "    " << k << " nearest neighbors:\n";
        for (std::size_t i = 0; i < k; ++i) {
            std::cout << "      ("
                      << neighbors[i].fx << ", " << neighbors[i].fy
                      << ")  class " << neighbors[i].label
                      << "  dist=" << std::setprecision(2)
                      << neighbors[i].dist << "\n";
        }

        // Count votes
        std::unordered_map<int, int> votes;
        for (std::size_t i = 0; i < k; ++i) {
            votes[neighbors[i].label]++;
        }

        std::cout << "    Votes: ";
        bool first = true;
        for (const auto& [label, count] : votes) {
            if (!first) std::cout << ", ";
            std::cout << "class " << label << "=" << count;
            first = false;
        }
        std::cout << "\n";

        std::cout << "    Prediction: " << class_names[prediction] << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
