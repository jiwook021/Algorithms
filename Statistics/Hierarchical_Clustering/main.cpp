/**
 * @file main.cpp
 * @brief Driver program for hierarchical clustering.
 */

#include <iostream>
#include <vector>

#include "HierarchicalClustering.hpp"

int main() {
    std::vector<Point> dataset = {
        {1.0, 2.0}, {1.5, 1.8}, {5.0, 8.0},
        {8.0, 8.0}, {1.0, 0.6}, {9.0, 11.0}
    };

    std::cout << "Starting hierarchical clustering...\n\n";

    auto clusters = HierarchicalClustering::Fit(dataset, 1);

    std::cout << "Final cluster contains " << clusters[0].Points.size()
              << " points:\n";
    for (const auto& p : clusters[0].Points) {
        std::cout << "  (" << p.X << ", " << p.Y << ")\n";
    }

    return 0;
}
