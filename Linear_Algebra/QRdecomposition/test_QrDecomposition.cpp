/**
 * @file test_QrDecomposition.cpp
 * @brief Google Test suite for QR decomposition (Modified Gram-Schmidt).
 * @details Verifies orthogonality of Q, upper triangularity of R,
 *          reconstruction A = Q*R, numerical stability on ill-conditioned
 *          matrices, error handling, and multi-type support.
 */

#include "QrDecomposition.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

// ============================================================================
// Helper utilities
// ============================================================================

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

/// Multiply two row-major matrices.
std::vector<std::vector<double>> Multiply(
        const std::vector<std::vector<double>>& a,
        const std::vector<std::vector<double>>& b) {
    const std::size_t m = a.size();
    const std::size_t n = b[0].size();
    const std::size_t k = b.size();
    std::vector<std::vector<double>> c(m, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t l = 0; l < k; ++l)
                c[i][j] += a[i][l] * b[l][j];
    return c;
}

/// Compute dot product of two columns from a row-major matrix.
double ColumnDot(const std::vector<std::vector<double>>& mat,
                 std::size_t col_a, std::size_t col_b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < mat.size(); ++i)
        sum += mat[i][col_a] * mat[i][col_b];
    return sum;
}

// ============================================================================
// Test: Orthogonality of Q
// ============================================================================

TEST(QrDecompositionTest, QColumnsAreOrthonormal) {
    PrintSection("Orthogonality of Q",
                 "Q^T Q must equal I -- columns are unit-length and mutually "
                 "orthogonal.  This is the defining property of an orthonormal "
                 "matrix and the primary output guarantee of Gram-Schmidt.");

    std::vector<std::vector<double>> a = {
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };
    auto result = QrDecomposition<double>::Decompose(a);

    const std::size_t n = result.Q[0].size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double dot = ColumnDot(result.Q, i, j);
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(dot, expected, 1e-9)
                << "Q^T Q [" << i << "][" << j << "] deviates from "
                << (i == j ? "1" : "0");
        }
    }
}

TEST(QrDecompositionTest, IdentityMatrixProducesOrthonormalQ) {
    PrintSection("Identity matrix",
                 "The identity is already orthonormal; Q should be +/-I and "
                 "R should be +/-I.  A basic smoke test.");

    auto result = QrDecomposition<double>::Decompose(
        {{1.0, 0.0}, {0.0, 1.0}});

    EXPECT_NEAR(std::abs(result.Q[0][0]), 1.0, 1e-12);
    EXPECT_NEAR(std::abs(result.Q[1][1]), 1.0, 1e-12);
    EXPECT_NEAR(result.R[0][0], 1.0, 1e-12);
    EXPECT_NEAR(result.R[1][1], 1.0, 1e-12);
    EXPECT_NEAR(result.R[1][0], 0.0, 1e-12);
}

// ============================================================================
// Test: R is upper triangular
// ============================================================================

TEST(QrDecompositionTest, RIsUpperTriangular) {
    PrintSection("Upper triangularity of R",
                 "All entries below the main diagonal of R must be zero.  "
                 "This is an algebraic invariant of QR decomposition.");

    std::vector<std::vector<double>> a = {
        {12.0, -51.0,   4.0},
        { 6.0, 167.0, -68.0},
        {-4.0,  24.0, -41.0}
    };
    auto result = QrDecomposition<double>::Decompose(a);

    const std::size_t n = result.R.size();
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            EXPECT_NEAR(result.R[i][j], 0.0, 1e-9)
                << "R[" << i << "][" << j << "] is not zero";
        }
    }
}

TEST(QrDecompositionTest, RIsUpperTriangularTall) {
    PrintSection("Upper triangularity (tall matrix)",
                 "For an m x n matrix with m > n the R factor is n x n.  "
                 "Verifying the invariant holds for non-square inputs.");

    std::vector<std::vector<double>> a = {
        {3.0, 1.0},
        {4.0, 1.0},
        {0.0, 1.0}
    };
    auto result = QrDecomposition<double>::Decompose(a);

    EXPECT_NEAR(result.R[1][0], 0.0, 1e-9);
}

// ============================================================================
// Test: Reconstruction  A = Q * R
// ============================================================================

TEST(QrDecompositionTest, ReconstructionAEqualsQR) {
    PrintSection("Reconstruction A = Q * R",
                 "The whole point of the decomposition: multiplying Q and R "
                 "back together must reproduce the original matrix to machine "
                 "precision.");

    std::vector<std::vector<double>> a = {
        {12.0, -51.0,   4.0},
        { 6.0, 167.0, -68.0},
        {-4.0,  24.0, -41.0}
    };
    auto result = QrDecomposition<double>::Decompose(a);
    auto product = Multiply(result.Q, result.R);

    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a[0].size(); ++j) {
            EXPECT_NEAR(product[i][j], a[i][j], 1e-9)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(QrDecompositionTest, ReconstructionSimple2x2) {
    PrintSection("Reconstruction (2x2)",
                 "Small matrix sanity check for A = Q * R.");

    std::vector<std::vector<double>> a = {{1.0, 1.0}, {0.0, 1.0}};
    auto result = QrDecomposition<double>::Decompose(a);
    auto product = Multiply(result.Q, result.R);

    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < a[0].size(); ++j)
            EXPECT_NEAR(product[i][j], a[i][j], 1e-9);
}

// ============================================================================
// Test: Numerical stability on ill-conditioned matrix
// ============================================================================

TEST(QrDecompositionTest, NumericalStabilityIllConditioned) {
    PrintSection("Numerical stability (ill-conditioned matrix)",
                 "Modified Gram-Schmidt maintains orthogonality much better "
                 "than Classical GS on matrices with high condition number.  "
                 "We test with a near-singular matrix and verify Q^T Q ~ I.");

    std::vector<std::vector<double>> a = {
        {1.0,       1.0,       1.0},
        {1e-4,      1e-4,      0.0},
        {1e-4,      0.0,       1e-4}
    };

    auto result = QrDecomposition<double>::Decompose(a);

    const std::size_t n = result.Q[0].size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double dot = ColumnDot(result.Q, i, j);
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(dot, expected, 1e-6)
                << "Q^T Q [" << i << "][" << j
                << "] on ill-conditioned input";
        }
    }

    auto product = Multiply(result.Q, result.R);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < a[0].size(); ++j)
            EXPECT_NEAR(product[i][j], a[i][j], 1e-9)
                << "Reconstruction mismatch at (" << i << ", " << j << ")";
}

TEST(QrDecompositionTest, NumericalStabilityHilbert) {
    PrintSection("Numerical stability (Hilbert matrix)",
                 "The Hilbert matrix H(i,j) = 1/(i+j+1) is notoriously "
                 "ill-conditioned.  A 4x4 Hilbert matrix has kappa ~ 1.5e4. "
                 "MGS should still produce a nearly orthonormal Q.");

    const std::size_t n = 4;
    std::vector<std::vector<double>> h(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            h[i][j] = 1.0 / static_cast<double>(i + j + 1);

    auto result = QrDecomposition<double>::Decompose(h);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double dot = ColumnDot(result.Q, i, j);
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(dot, expected, 1e-8)
                << "Q^T Q [" << i << "][" << j << "] on Hilbert matrix";
        }
    }

    auto product = Multiply(result.Q, result.R);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            EXPECT_NEAR(product[i][j], h[i][j], 1e-9);
}

// ============================================================================
// Test: Error handling
// ============================================================================

TEST(QrDecompositionTest, ThrowsOnEmptyMatrix) {
    PrintSection("Error handling: empty matrix",
                 "Decompose must reject an empty matrix with "
                 "std::invalid_argument.");

    EXPECT_THROW(QrDecomposition<double>::Decompose({}),
                 std::invalid_argument);
}

TEST(QrDecompositionTest, ThrowsOnLinearlyDependentColumns) {
    PrintSection("Error handling: linearly dependent columns",
                 "If columns are dependent, the Gram-Schmidt norm drops to "
                 "zero.  Decompose must throw std::runtime_error.");

    std::vector<std::vector<double>> a = {{1.0, 2.0}, {2.0, 4.0}};
    EXPECT_THROW(QrDecomposition<double>::Decompose(a),
                 std::runtime_error);
}

// ============================================================================
// Test: Multi-type instantiation (float, long double)
// ============================================================================

TEST(QrDecompositionTest, FloatInstantiation) {
    PrintSection("Float instantiation",
                 "Verify that QrDecomposition<float> compiles and produces "
                 "correct results within single-precision tolerance.");

    std::vector<std::vector<float>> a = {
        {1.0f, 1.0f},
        {0.0f, 1.0f}
    };
    auto result = QrDecomposition<float>::Decompose(a);

    // Reconstruction check.
    float product_00 = result.Q[0][0] * result.R[0][0] +
                       result.Q[0][1] * result.R[1][0];
    float product_01 = result.Q[0][0] * result.R[0][1] +
                       result.Q[0][1] * result.R[1][1];
    EXPECT_NEAR(product_00, 1.0f, 1e-5f);
    EXPECT_NEAR(product_01, 1.0f, 1e-5f);
}

TEST(QrDecompositionTest, LongDoubleInstantiation) {
    PrintSection("Long double instantiation",
                 "Verify that QrDecomposition<long double> compiles and "
                 "produces correct results.");

    std::vector<std::vector<long double>> a = {
        {1.0L, 1.0L, 0.0L},
        {1.0L, 0.0L, 1.0L},
        {0.0L, 1.0L, 1.0L}
    };
    auto result = QrDecomposition<long double>::Decompose(a);

    const std::size_t n = result.Q[0].size();
    for (std::size_t i = 0; i < n; ++i) {
        long double norm = 0.0L;
        for (std::size_t k = 0; k < result.Q.size(); ++k)
            norm += result.Q[k][i] * result.Q[k][i];
        EXPECT_NEAR(static_cast<double>(norm), 1.0, 1e-12)
            << "Column " << i << " of Q is not unit-length";
    }
}

}  // namespace
