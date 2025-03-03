/**
 * @file test_BasicStatistics.cpp
 * @brief Unit tests for the BasicStatistics class.
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "BasicStatistics.hpp"

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_NEAR(a, b, tol, msg)                                  \
    do {                                                             \
        if (std::fabs((a) - (b)) > (tol)) {                         \
            std::cerr << "FAIL: " << msg << " (expected " << (b)    \
                      << ", got " << (a) << ")\n";                   \
            testsFailed++;                                           \
        } else {                                                     \
            std::cout << "PASS: " << msg << "\n";                    \
            testsPassed++;                                           \
        }                                                            \
    } while (0)

/**
 * @brief Mean of {1,2,3,4,5} should be 3.0.
 */
void TestMean() {
    double result = BasicStatistics::Mean({1.0, 2.0, 3.0, 4.0, 5.0});
    ASSERT_NEAR(result, 3.0, 1e-10, "Mean of {1,2,3,4,5}");
}

/**
 * @brief Mean of a single element.
 */
void TestMeanSingle() {
    double result = BasicStatistics::Mean({7.0});
    ASSERT_NEAR(result, 7.0, 1e-10, "Mean of {7.0}");
}

/**
 * @brief Slope of a perfect y = 2x line should be 2.0.
 */
void TestSlope() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    double mx = BasicStatistics::Mean(x);
    double my = BasicStatistics::Mean(y);
    double m = BasicStatistics::Slope(x, y, mx, my);
    ASSERT_NEAR(m, 2.0, 1e-10, "Slope of y = 2x data");
}

/**
 * @brief Intercept when meanY=6, m=2, meanX=3 should be 0.
 */
void TestIntercept() {
    double b = BasicStatistics::Intercept(6.0, 2.0, 3.0);
    ASSERT_NEAR(b, 0.0, 1e-10, "Intercept(6, 2, 3)");
}

/**
 * @brief Predict(5, 2, 1) = 11.
 */
void TestPredict() {
    double result = BasicStatistics::Predict(5.0, 2.0, 1.0);
    ASSERT_NEAR(result, 11.0, 1e-10, "Predict(5, 2, 1)");
}

/**
 * @brief MSE of a perfect fit should be 0.
 */
void TestMsePerfect() {
    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {3, 5, 7};
    double mse = BasicStatistics::MeanSquaredError(x, y, 2.0, 1.0);
    ASSERT_NEAR(mse, 0.0, 1e-10, "MSE of perfect fit y = 2x + 1");
}

/**
 * @brief MSE of an imperfect fit should be positive but small.
 */
void TestMseNonZero() {
    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {3, 5, 8};
    double mx = BasicStatistics::Mean(x);
    double my = BasicStatistics::Mean(y);
    double m = BasicStatistics::Slope(x, y, mx, my);
    double b = BasicStatistics::Intercept(my, m, mx);
    double mse = BasicStatistics::MeanSquaredError(x, y, m, b);
    bool inRange = (mse > 0.0) && (mse < 1.0);
    if (inRange) {
        std::cout << "PASS: MSE of imperfect fit is positive and < 1\n";
        testsPassed++;
    } else {
        std::cerr << "FAIL: MSE of imperfect fit out of range: " << mse << "\n";
        testsFailed++;
    }
}

/**
 * @brief Slope of a horizontal line should be 0.
 */
void TestSlopeHorizontal() {
    std::vector<double> x = {1, 2, 3, 4};
    std::vector<double> y = {5, 5, 5, 5};
    double mx = BasicStatistics::Mean(x);
    double my = BasicStatistics::Mean(y);
    double m = BasicStatistics::Slope(x, y, mx, my);
    ASSERT_NEAR(m, 0.0, 1e-10, "Slope of horizontal line");
}

int main() {
    std::cout << "=== BasicStatistics Unit Tests ===\n\n";

    TestMean();
    TestMeanSingle();
    TestSlope();
    TestIntercept();
    TestPredict();
    TestMsePerfect();
    TestMseNonZero();
    TestSlopeHorizontal();

    std::cout << "\n=== Results: " << testsPassed << " passed, "
              << testsFailed << " failed ===\n";
    return testsFailed > 0 ? 1 : 0;
}
