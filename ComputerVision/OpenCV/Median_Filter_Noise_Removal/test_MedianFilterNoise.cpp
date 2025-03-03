/**
 * @file test_MedianFilterNoise.cpp
 * @brief Unit tests for MedianFilterNoise module
 */

#include "MedianFilterNoise.hpp"

// The tests are already embedded in MedianFilterNoise.hpp via gtest.
// This file provides a standalone test runner.

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
