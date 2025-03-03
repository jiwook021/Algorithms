/**
 * @file main.cpp
 * @brief Driver for NeuralNetworkImplementation -- linear regression via GD.
 */
#include "NeuralNetworkImplementation.hpp"
#include <iostream>

int main() {
    using namespace Ml;
    Matrix x(3, 2);
    x(0,0)=1; x(0,1)=2; x(1,0)=3; x(1,1)=4; x(2,0)=5; x(2,1)=6;
    Vector y({3.5, 7.5, 11.5});

    Vector w = LinearRegressionGD(x, y, 0.01, 1000);
    std::cout << "Learned weights: " << w[0] << ", " << w[1] << "\n";
    return 0;
}
