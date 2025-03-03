/**
 * @file main.cpp
 * @brief Driver for StlAlgorithm
 */

#include "StlAlgorithm.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace StlAlgorithm;

template <typename T>
void PrintVector(const std::vector<T>& vec) {
    for (const auto& v : vec) std::cout << v << " ";
    std::cout << std::endl;
}

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6};

    std::cout << "Original vector: ";
    PrintVector(vec);

    auto isEven = [](int x) { return x % 2 == 0; };

    std::cout << "MyAnyOf (even?): "
              << (MyAnyOf(vec.begin(), vec.end(), isEven) ? "Yes" : "No")
              << std::endl;

    std::cout << "MyNoneOf (negative?): "
              << (MyNoneOf(vec.begin(), vec.end(), [](int x) { return x < 0; }) ? "Yes" : "No")
              << std::endl;

    std::cout << "MyCountIf (odd): "
              << MyCountIf(vec.begin(), vec.end(), [](int x) { return x % 2 != 0; })
              << std::endl;

    return 0;
}
