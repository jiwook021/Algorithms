/**
 * @file main.cpp
 * @brief Driver program demonstrating the Xtensor array library.
 */

#include <iostream>
#include "Xtensor.hpp"

int main() {
    using namespace Xt;

    auto z = Zeros<double>({3, 4});
    auto o = Ones<double>({2, 5});

    std::cout << "zeros({3,4}) first element: " << z({0, 0}) << "\n";
    std::cout << "ones({2,5}) first element: " << o({0, 0}) << "\n";

    auto ls = Linspace<double>(0.0, 1.0, 5);
    std::cout << "linspace(0,1,5): ";
    for (auto val : ls.Data) std::cout << val << " ";
    std::cout << "\n";

    auto ar = Arange<int>(0, 10, 2);
    std::cout << "arange(0,10,2): ";
    for (auto val : ar.Data) std::cout << val << " ";
    std::cout << "\n";

    auto eye = Eye<double>(4);
    std::cout << "eye(4):\n";
    for (size_t i = 0; i < eye.Shape[0]; ++i) {
        for (size_t j = 0; j < eye.Shape[1]; ++j) {
            std::cout << eye({i, j}) << " ";
        }
        std::cout << "\n";
    }

    auto arr1d = Arange<int>(0, 12, 1);
    auto reshaped = Reshape(arr1d, {3, 4});
    std::cout << "Reshaped (3x4):\n";
    for (size_t i = 0; i < reshaped.Shape[0]; ++i) {
        for (size_t j = 0; j < reshaped.Shape[1]; ++j) {
            std::cout << reshaped({i, j}) << "\t";
        }
        std::cout << "\n";
    }

    auto transposed = Transpose(reshaped);
    std::cout << "Transposed (4x3):\n";
    for (size_t i = 0; i < transposed.Shape[0]; ++i) {
        for (size_t j = 0; j < transposed.Shape[1]; ++j) {
            std::cout << transposed({i, j}) << "\t";
        }
        std::cout << "\n";
    }

    return 0;
}
