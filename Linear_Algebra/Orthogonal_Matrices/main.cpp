/**
 * @file main.cpp
 * @brief Driver for Gram-Schmidt orthonormalization and orthogonality check.
 */

#include "OrthogonalMatrices.hpp"

#include <cstdlib>
#include <iostream>

int main() {
    try {
        using Matrix = OrthogonalMatrices::Matrix;

        std::cout << "Gram-Schmidt Orthonormalization:\n";
        Matrix inputVectors = {
            {1.0, 1.0, 0.0},
            {1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0}
        };

        auto basis = OrthogonalMatrices::GramSchmidt(inputVectors);
        for (const auto& v : basis) {
            for (double val : v) std::cout << "  " << val;
            std::cout << "\n";
        }

        std::cout << "\nOrthogonal Matrix Check:\n";
        Matrix test = {
            {1.0,  0.0, 0.0},
            {0.0,  0.0, -1.0},
            {0.0,  1.0, 0.0}
        };
        std::cout << (OrthogonalMatrices::IsOrthogonalMatrix(test)
                          ? "Orthogonal"
                          : "NOT orthogonal")
                  << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
