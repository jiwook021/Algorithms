/**
 * @file test_LinearRegression.cpp
 * @brief Unit tests for LinearRegression.
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "LinearRegression.hpp"

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
 * @brief Perfect linear data y = 2x + 1 should recover exact slope and intercept.
 */
void TestPerfectFit() {
    std::vector<DataPoint> data = {{1, 3}, {2, 5}, {3, 7}, {4, 9}};
    LinearRegression lr;
    lr.Fit(data);
    ASSERT_NEAR(lr.GetSlope(), 2.0, 1e-10, "Slope of y=2x+1 data");
    ASSERT_NEAR(lr.GetIntercept(), 1.0, 1e-10, "Intercept of y=2x+1 data");
}

/**
 * @brief Prediction should follow y = mx + b.
 */
void TestPredict() {
    std::vector<DataPoint> data = {{0, 1}, {1, 3}, {2, 5}};
    LinearRegression lr;
    lr.Fit(data);
    ASSERT_NEAR(lr.Predict(3), 7.0, 1e-10, "Predict(3) for y=2x+1");
}

/**
 * @brief Horizontal line should have slope 0.
 */
void TestHorizontalLine() {
    std::vector<DataPoint> data = {{1, 5}, {2, 5}, {3, 5}};
    LinearRegression lr;
    lr.Fit(data);
    ASSERT_NEAR(lr.GetSlope(), 0.0, 1e-10, "Slope of horizontal line");
    ASSERT_NEAR(lr.GetIntercept(), 5.0, 1e-10, "Intercept of y=5 line");
}

/**
 * @brief Negative slope should be detected.
 */
void TestNegativeSlope() {
    std::vector<DataPoint> data = {{0, 10}, {5, 0}};
    LinearRegression lr;
    lr.Fit(data);
    ASSERT_NEAR(lr.GetSlope(), -2.0, 1e-10, "Negative slope");
}

/**
 * @brief Two-point dataset should fit exactly through both points.
 */
void TestTwoPoints() {
    std::vector<DataPoint> data = {{0, 0}, {1, 1}};
    LinearRegression lr;
    lr.Fit(data);
    ASSERT_NEAR(lr.GetSlope(), 1.0, 1e-10, "Slope through (0,0) and (1,1)");
    ASSERT_NEAR(lr.GetIntercept(), 0.0, 1e-10, "Intercept through origin");
}

int main() {
    std::cout << "=== LinearRegression Unit Tests ===\n\n";

    TestPerfectFit();
    TestPredict();
    TestHorizontalLine();
    TestNegativeSlope();
    TestTwoPoints();

    std::cout << "\n=== Results: " << testsPassed << " passed, "
              << testsFailed << " failed ===\n";
    return testsFailed > 0 ? 1 : 0;
}
