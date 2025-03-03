/**
 * @file main.cpp
 * @brief Driver for Iterators (NumIterator and NumRange)
 */

#include "Iterators.hpp"
#include <iostream>

int main() {
    std::cout << "NumRange(100, 110): ";
    for (int i : NumRange(100, 110)) {
        std::cout << i << " ";
    }
    std::cout << "\n";

    std::cout << "NumRange(1, 6): ";
    for (int i : NumRange(1, 6)) {
        std::cout << i << " ";
    }
    std::cout << "\n";

    return 0;
}
