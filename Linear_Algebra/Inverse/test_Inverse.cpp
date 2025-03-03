/**
 * @file test_Inverse.cpp
 * @brief Google Test suite for templatized Gauss-Jordan matrix inversion.
 * @details Verifies A * A^{-1} = I, singular matrix detection, identity
 *          inverse, and multi-type support (float / double / long double).
 */

#include "Inverse.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// ============================================================================
// Helper utilities
// ============================================================================

namespace {

/// Print an educational section header before each logical group of tests.
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

/// Multiply two matrices (raw 2-D vectors).
template <typename Scalar>
std::vector<std::vector<Scalar>> Multiply(
    const Matrix<Scalar>& a, const Matrix<Scalar>& b) {
    const std::size_t m = a.Rows();
    const std::size_t n = b.Cols();
    const std::size_t k = a.Cols();
    std::vector<std::vector<Scalar>> c(m, std::vector<Scalar>(n,
                                       static_cast<Scalar>(0)));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t l = 0; l < k; ++l) {
                c[i][j] += a.At(i, l) * b.At(l, j);
            }
        }
    }
    return c;
}

/// Assert that a 2-D vector is the identity matrix within tolerance.
template <typename Scalar>
void ExpectIdentity(const std::vector<std::vector<Scalar>>& p,
                    Scalar tol = static_cast<Scalar>(1e-9)) {
    const std::size_t n = p.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Scalar expected = (i == j) ? static_cast<Scalar>(1)
                                       : static_cast<Scalar>(0);
            EXPECT_NEAR(static_cast<double>(p[i][j]),
                        static_cast<double>(expected),
                        static_cast<double>(tol))
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

}  // namespace

// ============================================================================
// Test: A * A^{-1} = I  (the fundamental correctness property)
// ============================================================================

TEST(InverseTest, ProductIsIdentity2x2) {
    PrintSection("A * A^-1 = I (2x2)",
                 "The product of a matrix and its inverse must be the identity "
                 "matrix. This is the defining property of the matrix inverse.");

    Matrix<double> m({{3.0, 5.0}, {1.0, 2.0}});
    Matrix<double> inv = m.Inverse();
    auto product = Multiply(m, inv);
    ExpectIdentity(product);
}

TEST(InverseTest, ProductIsIdentity3x3) {
    PrintSection("A * A^-1 = I (3x3)",
                 "Gauss-Jordan elimination must handle larger matrices "
                 "correctly, including partial pivoting across three rows.");

    Matrix<double> m({{2.0, 1.0, 1.0},
                      {1.0, 3.0, 2.0},
                      {1.0, 0.0, 0.0}});
    Matrix<double> inv = m.Inverse();
    auto product = Multiply(m, inv);
    ExpectIdentity(product);
}

TEST(InverseTest, Known2x2Inverse) {
    PrintSection("Known 2x2 inverse",
                 "For [[4,7],[2,6]] with det=10 the closed-form inverse is "
                 "[[0.6,-0.7],[-0.2,0.4]]. Checking element-by-element.");

    Matrix<double> m({{4.0, 7.0}, {2.0, 6.0}});
    Matrix<double> inv = m.Inverse();

    EXPECT_NEAR(inv.At(0, 0),  0.6, 1e-9);
    EXPECT_NEAR(inv.At(0, 1), -0.7, 1e-9);
    EXPECT_NEAR(inv.At(1, 0), -0.2, 1e-9);
    EXPECT_NEAR(inv.At(1, 1),  0.4, 1e-9);
}

// ============================================================================
// Test: singular matrix throws
// ============================================================================

TEST(InverseTest, SingularMatrixThrows) {
    PrintSection("Singular matrix detection",
                 "A matrix with linearly dependent rows (det=0) has no "
                 "inverse. The algorithm must detect a near-zero pivot and "
                 "throw std::runtime_error(\"Matrix is singular\").");

    Matrix<double> m({{1.0, 2.0}, {2.0, 4.0}});
    EXPECT_THROW(m.Inverse(), std::runtime_error);
}

TEST(InverseTest, SingularMatrixMessage) {
    PrintSection("Singular error message",
                 "Verifying the exact error string so callers can rely on it.");

    Matrix<double> m({{1.0, 2.0}, {2.0, 4.0}});
    try {
        (void)m.Inverse();
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "Matrix is singular");
    }
}

TEST(InverseTest, NonSquareThrows) {
    PrintSection("Non-square matrix rejection",
                 "Inversion is defined only for square matrices. A 2x3 input "
                 "must be rejected immediately.");

    Matrix<double> m({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    EXPECT_THROW(m.Inverse(), std::runtime_error);
}

// ============================================================================
// Test: identity inverse is identity
// ============================================================================

TEST(InverseTest, IdentityInverseIsIdentity) {
    PrintSection("Identity inverse",
                 "The identity matrix is its own inverse: I^{-1} = I. "
                 "This is a basic sanity check for any inversion algorithm.");

    Matrix<double> id({{1.0, 0.0}, {0.0, 1.0}});
    Matrix<double> inv = id.Inverse();

    EXPECT_NEAR(inv.At(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(inv.At(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(inv.At(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(inv.At(1, 1), 1.0, 1e-12);
}

TEST(InverseTest, IdentityInverse3x3) {
    PrintSection("3x3 identity inverse",
                 "Checking I^{-1} = I for a 3x3 identity matrix.");

    Matrix<double> id({{1.0, 0.0, 0.0},
                       {0.0, 1.0, 0.0},
                       {0.0, 0.0, 1.0}});
    Matrix<double> inv = id.Inverse();

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(inv.At(i, j), expected, 1e-12);
        }
    }
}

// ============================================================================
// Test: 1x1 scalar inverse
// ============================================================================

TEST(InverseTest, SingleElement) {
    PrintSection("1x1 matrix inverse",
                 "A 1x1 matrix [[a]] has inverse [[1/a]]. Edge case for "
                 "the augmented-matrix loop.");

    Matrix<double> m({{5.0}});
    Matrix<double> inv = m.Inverse();
    EXPECT_NEAR(inv.At(0, 0), 0.2, 1e-12);
}

// ============================================================================
// Test: template works with float
// ============================================================================

TEST(InverseTest, FloatInstantiation) {
    PrintSection("Float instantiation",
                 "The template must compile and work correctly with float, "
                 "not just double. Tolerance is relaxed for float precision.");

    Matrix<float> m({{4.0f, 7.0f}, {2.0f, 6.0f}});
    Matrix<float> inv = m.Inverse();

    EXPECT_NEAR(static_cast<double>(inv.At(0, 0)),  0.6, 1e-5);
    EXPECT_NEAR(static_cast<double>(inv.At(0, 1)), -0.7, 1e-5);
    EXPECT_NEAR(static_cast<double>(inv.At(1, 0)), -0.2, 1e-5);
    EXPECT_NEAR(static_cast<double>(inv.At(1, 1)),  0.4, 1e-5);
}

// ============================================================================
// Test: dimension-based constructor
// ============================================================================

TEST(InverseTest, DimensionConstructor) {
    PrintSection("Dimension constructor",
                 "Matrix(rows, cols) must create a zero-initialized matrix "
                 "with correct dimensions.");

    Matrix<double> m(3, 4);
    EXPECT_EQ(m.Rows(), 3u);
    EXPECT_EQ(m.Cols(), 4u);
    EXPECT_NEAR(m.At(0, 0), 0.0, 1e-15);
    EXPECT_NEAR(m.At(2, 3), 0.0, 1e-15);
}

TEST(InverseTest, MutableAccess) {
    PrintSection("Mutable element access",
                 "At(row, col) must return a mutable reference so callers "
                 "can build matrices element by element.");

    Matrix<double> m(2, 2);
    m.At(0, 0) = 4.0;
    m.At(0, 1) = 7.0;
    m.At(1, 0) = 2.0;
    m.At(1, 1) = 6.0;

    Matrix<double> inv = m.Inverse();
    EXPECT_NEAR(inv.At(0, 0),  0.6, 1e-9);
    EXPECT_NEAR(inv.At(0, 1), -0.7, 1e-9);
}
