/**
 * @file main.cpp
 * @brief Driver for PolynomialCalculator
 */

#include "PolynomialCalculator.hpp"
#include <iostream>

using namespace PolynomialCalculator;

int main() {
    std::string input = "3x^2 + 2x - 1";

    std::string cleaned = RemoveSpaces(input);
    std::cout << "Cleaned: " << cleaned << std::endl;

    auto terms = SplitPolynomial(cleaned);
    std::cout << "Terms:" << std::endl;
    for (const auto& t : terms) {
        std::cout << "  [" << t << "]" << std::endl;
    }

    return 0;
}
