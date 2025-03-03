/**
 * @file main.cpp
 * @brief Driver for FeatureDeployment
 */

#include "FeatureDeployment.hpp"
#include <iostream>

int main() {
    auto result = FeatureDeployment::Solution({93, 30, 55}, {1, 30, 5});

    std::cout << "Deployment groups: ";
    for (int v : result) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    return 0;
}
