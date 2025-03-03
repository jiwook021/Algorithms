/**
 * @file RandomGenerator.cpp
 * @brief Implementation of random number generation and counting
 */

#include "RandomGenerator.hpp"

namespace RandomGenerator {

std::vector<int> Generate(int lowerBound, int upperBound, std::size_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(lowerBound, upperBound);

    std::vector<int> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        result.push_back(dist(gen));
    }
    return result;
}

void CountOccurrences(const std::vector<int>& values,
                      std::array<uint16_t, RESULT_ARRAY_SIZE>& result) {
    result.fill(0);
    for (int v : values) {
        if (v >= 0 && v < RESULT_ARRAY_SIZE) {
            result[v]++;
        }
    }
}

} // namespace RandomGenerator
