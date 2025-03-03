/**
 * @file test_Xtensor.cpp
 * @brief Unit tests for the Xtensor library.
 */

#include <cmath>
#include <iostream>

#include "Xtensor.hpp"

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
#define ASSERT_NEAR(a, b, tol, msg) ASSERT_TRUE(std::fabs((a)-(b)) < (tol), msg)

/**
 * @brief Zeros creates an array filled with 0.
 */
void TestZeros() {
    auto arr = Xt::Zeros<double>({2, 3});
    ASSERT_EQ(arr.Size(), 6u, "Zeros({2,3}) has 6 elements");
    bool allZero = true;
    for (auto& v : arr.Data) {
        if (v != 0.0) allZero = false;
    }
    ASSERT_TRUE(allZero, "Zeros: all elements are 0");
}

/**
 * @brief Ones creates an array filled with 1.
 */
void TestOnes() {
    auto arr = Xt::Ones<int>({3, 2});
    bool allOne = true;
    for (auto& v : arr.Data) {
        if (v != 1) allOne = false;
    }
    ASSERT_TRUE(allOne, "Ones: all elements are 1");
}

/**
 * @brief Element access read/write.
 */
void TestElementAccess() {
    auto arr = Xt::Zeros<double>({2, 3});
    arr({0, 1}) = 5.0;
    arr({1, 2}) = 7.0;
    ASSERT_NEAR(arr({0, 1}), 5.0, 1e-10, "Read back element (0,1)=5.0");
    ASSERT_NEAR(arr({1, 2}), 7.0, 1e-10, "Read back element (1,2)=7.0");
    ASSERT_NEAR(arr({0, 0}), 0.0, 1e-10, "Unset element (0,0)=0.0");
}

/**
 * @brief Eye creates an identity matrix.
 */
void TestEye() {
    auto eye = Xt::Eye<double>(3);
    ASSERT_NEAR(eye({0, 0}), 1.0, 1e-10, "Eye(3) diagonal (0,0)=1");
    ASSERT_NEAR(eye({1, 1}), 1.0, 1e-10, "Eye(3) diagonal (1,1)=1");
    ASSERT_NEAR(eye({2, 2}), 1.0, 1e-10, "Eye(3) diagonal (2,2)=1");
    ASSERT_NEAR(eye({0, 1}), 0.0, 1e-10, "Eye(3) off-diagonal (0,1)=0");
}

/**
 * @brief Linspace produces evenly spaced values.
 */
void TestLinspace() {
    auto arr = Xt::Linspace<double>(0.0, 1.0, 5);
    ASSERT_EQ(arr.Size(), 5u, "Linspace(0,1,5) has 5 elements");
    ASSERT_NEAR(arr.Data[0], 0.0, 1e-10, "Linspace first = 0.0");
    ASSERT_NEAR(arr.Data[4], 1.0, 1e-10, "Linspace last = 1.0");
    ASSERT_NEAR(arr.Data[2], 0.5, 1e-10, "Linspace middle = 0.5");
}

/**
 * @brief 3D array has correct size.
 */
void TestShape3D() {
    auto arr = Xt::Zeros<float>({2, 3, 4});
    ASSERT_EQ(arr.Size(), 24u, "3D array {2,3,4} has 24 elements");
    ASSERT_EQ(arr.Shape.size(), 3u, "3D array has 3 dimensions");
}

/**
 * @brief Arange produces correct sequence.
 */
void TestArange() {
    auto arr = Xt::Arange<int>(0, 6, 2);
    ASSERT_EQ(arr.Size(), 3u, "Arange(0,6,2) has 3 elements");
    ASSERT_EQ(arr.Data[0], 0, "Arange first = 0");
    ASSERT_EQ(arr.Data[1], 2, "Arange second = 2");
    ASSERT_EQ(arr.Data[2], 4, "Arange third = 4");
}

/**
 * @brief Reshape preserves data and changes shape.
 */
void TestReshape() {
    auto arr = Xt::Arange<int>(0, 6, 1);
    auto reshaped = Xt::Reshape(arr, {2, 3});
    ASSERT_EQ(reshaped.Shape[0], 2u, "Reshaped rows = 2");
    ASSERT_EQ(reshaped.Shape[1], 3u, "Reshaped cols = 3");
    ASSERT_EQ(reshaped({1, 0}), 3, "Reshaped (1,0) = 3");
}

/**
 * @brief Transpose swaps dimensions.
 */
void TestTranspose() {
    auto arr = Xt::Zeros<int>({2, 3});
    arr({0, 1}) = 5;
    arr({1, 2}) = 7;
    auto t = Xt::Transpose(arr);
    ASSERT_EQ(t.Shape[0], 3u, "Transposed rows = 3");
    ASSERT_EQ(t.Shape[1], 2u, "Transposed cols = 2");
    ASSERT_EQ(t({1, 0}), 5, "Transposed (1,0) = original (0,1) = 5");
    ASSERT_EQ(t({2, 1}), 7, "Transposed (2,1) = original (1,2) = 7");
}

int main() {
    std::cout << "=== Xtensor Unit Tests ===\n\n";

    TestZeros();
    TestOnes();
    TestElementAccess();
    TestEye();
    TestLinspace();
    TestShape3D();
    TestArange();
    TestReshape();
    TestTranspose();

    std::cout << "\n=== Results: " << testsPassed << " passed, "
              << testsFailed << " failed ===\n";
    return testsFailed > 0 ? 1 : 0;
}
