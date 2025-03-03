/**
 * @file test_MaxPooling.cu
 * @brief Unit tests for the MaxPooling CUDA kernel.
 */

#include "MaxPooling.cuh"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_EQ(actual, expected, msg)                                      \
    do {                                                                      \
        if (std::abs((actual) - (expected)) > 1e-5f) {                        \
            std::cerr << "[FAIL] " << (msg)                                   \
                      << ": expected " << (expected)                          \
                      << ", got " << (actual) << std::endl;                   \
            testsFailed++;                                                    \
        } else {                                                              \
            testsPassed++;                                                    \
        }                                                                     \
    } while (0)

void Test2x2PoolOn4x4() {
    std::vector<float> input = {
        1, 2, 5, 4,
        5, 6, 7, 8,
        3, 2, 1, 0,
        1, 2, 3, 4
    };
    std::vector<float> output;
    MaxPooling(input, output, 4, 4, 2, 2);

    std::vector<float> expected = {6, 8, 3, 4};
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(output[i], expected[i], "2x2 pool on 4x4");
    }
    std::cout << "[PASSED] Test2x2PoolOn4x4" << std::endl;
}

void Test2x2PoolOn5x5NonDivisible() {
    std::vector<float> input = {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    std::vector<float> output;
    MaxPooling(input, output, 5, 5, 2, 2);

    std::vector<float> expected = {7, 9, 10, 17, 19, 20, 22, 24, 25};
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(output[i], expected[i], "2x2 pool on 5x5");
    }
    std::cout << "[PASSED] Test2x2PoolOn5x5NonDivisible" << std::endl;
}

void Test3x3PoolOn6x6() {
    std::vector<float> input = {
         1,  2,  3,  4,  5,  6,
         7,  8,  9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36
    };
    std::vector<float> output;
    MaxPooling(input, output, 6, 6, 3, 3);

    std::vector<float> expected = {15, 18, 33, 36};
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(output[i], expected[i], "3x3 pool on 6x6");
    }
    std::cout << "[PASSED] Test3x3PoolOn6x6" << std::endl;
}

void Test1x1Pool() {
    std::vector<float> input = {1, 2, 3, 4};
    std::vector<float> output;
    MaxPooling(input, output, 2, 2, 1, 1);

    std::vector<float> expected = {1, 2, 3, 4};
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(output[i], expected[i], "1x1 pool (identity)");
    }
    std::cout << "[PASSED] Test1x1Pool" << std::endl;
}

void TestNegativeValues() {
    std::vector<float> input = {
        -5, -3,
        -1, -8
    };
    std::vector<float> output;
    MaxPooling(input, output, 2, 2, 2, 2);

    ASSERT_EQ(output[0], -1.0f, "negative values pool");
    std::cout << "[PASSED] TestNegativeValues" << std::endl;
}

int main() {
    Test2x2PoolOn4x4();
    Test2x2PoolOn5x5NonDivisible();
    Test3x3PoolOn6x6();
    Test1x1Pool();
    TestNegativeValues();

    std::cout << "\n" << testsPassed << " assertions passed, "
              << testsFailed << " failed." << std::endl;
    return testsFailed == 0 ? 0 : 1;
}
