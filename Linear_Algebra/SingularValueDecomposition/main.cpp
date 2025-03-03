/**
 * @file main.cpp
 * @brief Driver program for Singular Value Decomposition.
 */

#include "SingularValueDecomposition.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>

int main() {
    try {
        using SVD = SingularValueDecomposition<double>;
        SVD::Matrix a = {
            {3.0, 2.0,  2.0},
            {2.0, 3.0, -2.0}
        };

        auto result = SVD::Decompose(a);

        std::cout << "Singular values:";
        for (double sv : result.SingularValues)
            std::cout << "  " << sv;
        std::cout << "\n\nRank:             "
                  << SVD::Rank(result) << "\n";
        std::cout << "Condition number: "
                  << SVD::ConditionNumber(result) << "\n";
        std::cout << "Frobenius norm:   "
                  << SVD::FrobeniusNorm(result) << "\n";

        auto reconstructed = SVD::Reconstruct(result);
        double err = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < a[0].size(); ++j)
                err += (a[i][j] - reconstructed[i][j]) *
                       (a[i][j] - reconstructed[i][j]);
        std::cout << "Reconstruction error: " << std::sqrt(err) << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
