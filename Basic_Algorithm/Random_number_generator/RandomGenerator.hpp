/**
 * @file RandomGenerator.hpp
 * @brief Random number generation and occurrence counting
 * @details Generates random integers in a range using Mersenne Twister
 *          and counts occurrences of each value.
 */

#pragma once

#include <vector>
#include <array>
#include <cstdint>
#include <random>

namespace RandomGenerator {

static constexpr int RESULT_ARRAY_SIZE = 256;

/**
 * @brief Generate random integers in [lowerBound, upperBound].
 * @param lowerBound Minimum value (inclusive)
 * @param upperBound Maximum value (inclusive)
 * @param count      Number of random values to generate
 * @return Vector of random integers
 */
std::vector<int> Generate(int lowerBound, int upperBound, std::size_t count);

/**
 * @brief Count occurrences of each value in the vector.
 * @param values Input values (must be in [0, RESULT_ARRAY_SIZE))
 * @param result Output array of counts (zeroed first)
 */
void CountOccurrences(const std::vector<int>& values,
                      std::array<uint16_t, RESULT_ARRAY_SIZE>& result);

} // namespace RandomGenerator
