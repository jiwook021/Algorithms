/**
 * @file main.cpp
 * @brief Driver program demonstrating p-norm computation.
 */

#include "Norm.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    try {
        std::vector<double> vec{3.0, -4.0, 5.0, -2.0};

        std::cout << "L1 Norm:       " << Norm::L1(vec) << "\n";
        std::cout << "L2 Norm:       " << Norm::L2(vec) << "\n";
        std::cout << "L3 Norm:       "
                  << Norm::PNorm(vec, 3.0) << "\n";
        std::cout << "Infinity Norm: " << Norm::LInf(vec) << "\n";

        std::vector<double> empty;
        std::cout << "Empty vector:  "
                  << Norm::PNorm(empty, 2.0) << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
