/**
 * @file main.cpp
 * @brief Driver for RandomGenerator
 */

#include "RandomGenerator.hpp"
#include <iostream>

int main() {
    auto numbers = RandomGenerator::Generate(0, 9, 100);

    std::cout << "Generated " << numbers.size() << " random numbers (0-9)." << std::endl;

    std::array<uint16_t, RandomGenerator::RESULT_ARRAY_SIZE> counts;
    RandomGenerator::CountOccurrences(numbers, counts);

    std::cout << "Occurrences:" << std::endl;
    for (int i = 0; i <= 9; ++i) {
        std::cout << "  [" << i << "]: " << counts[i] << std::endl;
    }

    return 0;
}
