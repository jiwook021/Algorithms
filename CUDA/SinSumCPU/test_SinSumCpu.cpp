/**
 * @file test_SinSumCpu.cpp
 * @brief Unit tests for SinSumCpu Taylor series computation.
 */

#include "SinSumCpu.hpp"
#include <cstdio>
#include <cmath>

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_NEAR(actual, expected, tol, msg)                                \
    do {                                                                       \
        if (std::abs((actual) - (expected)) > (tol)) {                        \
            fprintf(stderr, "[FAIL] %s: expected %.6f, got %.6f\n",           \
                    msg, (float)(expected), (float)(actual));                  \
            testsFailed++;                                                     \
        } else {                                                               \
            testsPassed++;                                                     \
        }                                                                      \
    } while (0)

void TestSinZero() {
    ASSERT_NEAR(SinSum(0.0f, 100), 0.0f, 1e-6f, "sin(0)");
    printf("[PASSED] TestSinZero\n");
}

void TestSinPiOver2() {
    ASSERT_NEAR(SinSum(3.14159265f / 2.0f, 100), 1.0f, 1e-5f, "sin(pi/2)");
    printf("[PASSED] TestSinPiOver2\n");
}

void TestSinPi() {
    ASSERT_NEAR(SinSum(3.14159265f, 100), 0.0f, 1e-5f, "sin(pi)");
    printf("[PASSED] TestSinPi\n");
}

void TestSinNegative() {
    float result = SinSum(-3.14159265f / 2.0f, 100);
    ASSERT_NEAR(result, -1.0f, 1e-5f, "sin(-pi/2)");
    printf("[PASSED] TestSinNegative\n");
}

void TestFewTerms() {
    // With 1 term, sin(x) ~ x
    float result = SinSum(0.1f, 1);
    ASSERT_NEAR(result, 0.1f, 1e-6f, "sin(0.1) with 1 term");
    printf("[PASSED] TestFewTerms\n");
}

int main() {
    TestSinZero();
    TestSinPiOver2();
    TestSinPi();
    TestSinNegative();
    TestFewTerms();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
