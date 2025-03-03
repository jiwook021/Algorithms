/**
 * @file test_RandomGenerator.cpp
 * @brief Unit tests for RandomGenerator
 */

#include "RandomGenerator.hpp"
#include <gtest/gtest.h>
#include <numeric>

using namespace RandomGenerator;

TEST(RandomGenerator, GeneratesCorrectCount) {
    auto numbers = Generate(1, 10, 100);
    EXPECT_EQ(numbers.size(), 100u);
}

TEST(RandomGenerator, ValuesInRange) {
    auto numbers = Generate(5, 15, 1000);
    for (int n : numbers) {
        EXPECT_GE(n, 5);
        EXPECT_LE(n, 15);
    }
}

TEST(RandomGenerator, ZeroCountReturnsEmpty) {
    auto numbers = Generate(0, 10, 0);
    EXPECT_TRUE(numbers.empty());
}

TEST(RandomGenerator, CountOccurrencesBasic) {
    std::vector<int> values = {1, 2, 2, 3, 3, 3};
    std::array<uint16_t, RESULT_ARRAY_SIZE> result;
    CountOccurrences(values, result);
    EXPECT_EQ(result[0], 0);
    EXPECT_EQ(result[1], 1);
    EXPECT_EQ(result[2], 2);
    EXPECT_EQ(result[3], 3);
}

TEST(RandomGenerator, CountOccurrencesSumsToTotal) {
    auto numbers = Generate(0, 9, 50);
    std::array<uint16_t, RESULT_ARRAY_SIZE> result;
    CountOccurrences(numbers, result);

    int total = 0;
    for (int i = 0; i < RESULT_ARRAY_SIZE; ++i) {
        total += result[i];
    }
    EXPECT_EQ(total, 50);
}

TEST(RandomGenerator, CountOccurrencesEmpty) {
    std::vector<int> empty;
    std::array<uint16_t, RESULT_ARRAY_SIZE> result;
    CountOccurrences(empty, result);
    for (int i = 0; i < RESULT_ARRAY_SIZE; ++i) {
        EXPECT_EQ(result[i], 0);
    }
}
