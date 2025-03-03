/**
 * @file main.cpp
 * @brief Driver program demonstrating symmetric matrix properties.
 */

#include "SymmetryMatrices.hpp"

#include <cstdlib>
#include <iostream>

int main() {
    try {
        std::cout << "=== Properties of Symmetric Matrices ===\n\n";

        SymmetryMatrix a(3);
        std::cout << "Random symmetric matrix A:\n";
        a.Print();
        std::cout << "Symmetric?  " << (a.IsSymmetric() ? "Yes" : "No") << "\n";
        std::cout << "Trace:      " << a.Trace() << "\n";
        std::cout << "Det:        " << a.Determinant() << "\n";
        std::cout << "Frobenius:  " << a.FrobeniusNorm() << "\n";
        std::cout << "Max eigenvalue (approx): "
                  << a.ApproximateLargestEigenvalue() << "\n\n";

        SymmetryMatrix id = CreateIdentityMatrix(3);
        std::cout << "A * I commutes? " << (a.IsCommutative(id) ? "Yes" : "No")
                  << "\n";
        std::cout << "A * I symmetric? "
                  << (a.IsProductSymmetric(id) ? "Yes" : "No") << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
