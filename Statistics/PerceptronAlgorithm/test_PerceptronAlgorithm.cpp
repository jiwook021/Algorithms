/**
 * @file test_PerceptronAlgorithm.cpp
 * @brief Unit tests for PerceptronAlgorithm.
 */

#include <iostream>
#include <vector>

#include "PerceptronAlgorithm.hpp"

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_TRUE(cond, msg)                                       \
    do {                                                             \
        if (!(cond)) {                                               \
            std::cerr << "FAIL: " << msg << "\n";                    \
            testsFailed++;                                           \
        } else {                                                     \
            std::cout << "PASS: " << msg << "\n";                    \
            testsPassed++;                                           \
        }                                                            \
    } while (0)

#define ASSERT_EQ(a, b, msg) ASSERT_TRUE((a) == (b), msg)

/**
 * @brief Step function: positive -> 1, negative -> 0, zero -> 1.
 */
void TestStepFunction() {
    PerceptronAlgorithm p;
    ASSERT_EQ(p.StepFunction(1.0), 1, "StepFunction(1.0) = 1");
    ASSERT_EQ(p.StepFunction(-1.0), 0, "StepFunction(-1.0) = 0");
    ASSERT_EQ(p.StepFunction(0.0), 1, "StepFunction(0.0) = 1");
}

/**
 * @brief Linearly separable data should be correctly classified.
 */
void TestLinearSeparable() {
    std::vector<DataPoint> data = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0},
        {2, 2, 1}, {3, 2, 1}, {2, 3, 1}
    };
    PerceptronAlgorithm p(0.1, 1000);
    p.Train(data);
    ASSERT_EQ(p.Predict(0, 0), 0, "Predict class 0 at origin");
    ASSERT_EQ(p.Predict(3, 3), 1, "Predict class 1 at (3,3)");
}

/**
 * @brief Training on two well-separated points should converge.
 */
void TestTrainingConverges() {
    std::vector<DataPoint> data = {{0, 0, 0}, {10, 10, 1}};
    PerceptronAlgorithm p(0.1, 100);
    p.Train(data);
    ASSERT_EQ(p.Predict(0, 0), 0, "Predict class 0 at origin");
    ASSERT_EQ(p.Predict(10, 10), 1, "Predict class 1 at (10,10)");
}

/**
 * @brief Training accuracy on separable data should be 100%.
 */
void TestFullAccuracy() {
    std::vector<DataPoint> data = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0},
        {5, 5, 1}, {6, 5, 1}, {5, 6, 1}
    };
    PerceptronAlgorithm p(0.1, 1000);
    p.Train(data);

    int correct = 0;
    for (const auto& dp : data) {
        if (p.Predict(dp.X1, dp.X2) == dp.Label) correct++;
    }
    ASSERT_EQ(correct, static_cast<int>(data.size()),
              "100% training accuracy on separable data");
}

int main() {
    std::cout << "=== PerceptronAlgorithm Unit Tests ===\n\n";

    TestStepFunction();
    TestLinearSeparable();
    TestTrainingConverges();
    TestFullAccuracy();

    std::cout << "\n=== Results: " << testsPassed << " passed, "
              << testsFailed << " failed ===\n";
    return testsFailed > 0 ? 1 : 0;
}
