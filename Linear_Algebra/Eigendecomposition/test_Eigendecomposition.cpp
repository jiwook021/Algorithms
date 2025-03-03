/**
 * @file test_Eigendecomposition.cpp
 * @brief Google Test suite for the Jacobi eigendecomposition algorithm.
 * @details Tests verify eigenvalue/eigenvector correctness by checking
 *          A * v == lambda * v for each eigenpair, orthogonality of the
 *          eigenvector matrix, symmetry validation, and edge-case handling.
 */

#include "Eigendecomposition.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

constexpr double kTestTolerance = 1e-8;

/// Multiply matrix A by vector v.
std::vector<double> MatVec(const std::vector<std::vector<double>>& a,
                           const std::vector<double>& v) {
    const auto n = a.size();
    std::vector<double> result(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[i] += a[i][j] * v[j];
        }
    }
    return result;
}

/// Verify A*v == lambda*v for every eigenpair.
void ExpectEigenpairsCorrect(
    const std::vector<std::vector<double>>& a,
    const Eigendecomposition<double>::EigenResult& result) {

    const auto n = a.size();
    for (std::size_t k = 0; k < n; ++k) {
        std::vector<double> v(n);
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = result.Eigenvectors[i][k];
        }
        auto av = MatVec(a, v);
        double lambda = result.Eigenvalues[k];
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(av[i], lambda * v[i], kTestTolerance)
                << "Eigenpair mismatch at eigenvalue index " << k
                << ", component " << i;
        }
    }
}

/// Verify the eigenvectors form an orthonormal set.
void ExpectOrthonormal(
    const Eigendecomposition<double>::EigenResult& result) {

    const auto n = result.Eigenvalues.size();
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = 0; b < n; ++b) {
            double dot = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                dot += result.Eigenvectors[i][a] *
                       result.Eigenvectors[i][b];
            }
            double expected = (a == b) ? 1.0 : 0.0;
            EXPECT_NEAR(dot, expected, kTestTolerance)
                << "Orthonormality failed for columns " << a
                << " and " << b;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class EigendecompositionTest : public ::testing::Test {};

// ---------------------------------------------------------------------------
// Symmetric decomposition: A*v == lambda*v
// ---------------------------------------------------------------------------

TEST_F(EigendecompositionTest, DiagonalMatrixEigenpairs) {
    PrintSection("Diagonal Matrix Eigenpairs",
                 "A diagonal matrix's eigenvalues are its diagonal entries; "
                 "eigenvectors should be the standard basis vectors.");

    std::vector<std::vector<double>> diag = {
        {5.0, 0.0, 0.0},
        {0.0, 3.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    auto result = Eigendecomposition<double>::Decompose(diag);

    // Eigenvalues should be {5, 3, 1} in some order.
    std::vector<double> sorted = result.Eigenvalues;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_NEAR(sorted[0], 1.0, kTestTolerance);
    EXPECT_NEAR(sorted[1], 3.0, kTestTolerance);
    EXPECT_NEAR(sorted[2], 5.0, kTestTolerance);

    ExpectEigenpairsCorrect(diag, result);
}

TEST_F(EigendecompositionTest, Symmetric3x3Eigenpairs) {
    PrintSection("3x3 Symmetric Matrix Eigenpairs",
                 "The Jacobi method must satisfy A*v = lambda*v for a "
                 "general symmetric matrix with off-diagonal entries.");

    std::vector<std::vector<double>> matrix = {
        { 4.0, -2.0,  2.0},
        {-2.0,  2.0, -4.0},
        { 2.0, -4.0, 11.0}
    };

    auto result = Eigendecomposition<double>::Decompose(matrix);
    ExpectEigenpairsCorrect(matrix, result);
}

TEST_F(EigendecompositionTest, IdentityMatrixEigenpairs) {
    PrintSection("Identity Matrix Eigenpairs",
                 "All eigenvalues of the identity matrix equal 1; "
                 "any orthonormal basis is a valid eigenvector set.");

    std::vector<std::vector<double>> id = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    auto result = Eigendecomposition<double>::Decompose(id);

    for (double val : result.Eigenvalues) {
        EXPECT_NEAR(val, 1.0, kTestTolerance);
    }
    ExpectEigenpairsCorrect(id, result);
}

TEST_F(EigendecompositionTest, SingleElementMatrix) {
    PrintSection("1x1 Matrix Eigenpair",
                 "Trivial case: the sole entry is the eigenvalue and "
                 "the eigenvector is [1].");

    std::vector<std::vector<double>> single = {{7.0}};
    auto result = Eigendecomposition<double>::Decompose(single);

    EXPECT_NEAR(result.Eigenvalues[0], 7.0, kTestTolerance);
    EXPECT_NEAR(result.Eigenvectors[0][0], 1.0, kTestTolerance);
}

// ---------------------------------------------------------------------------
// Eigenvector orthogonality
// ---------------------------------------------------------------------------

TEST_F(EigendecompositionTest, EigenvectorOrthogonalityDiagonal) {
    PrintSection("Eigenvector Orthogonality (Diagonal)",
                 "For a real symmetric matrix the eigenvectors must form "
                 "an orthonormal set (V^T V = I).");

    std::vector<std::vector<double>> diag = {
        {5.0, 0.0, 0.0},
        {0.0, 3.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    auto result = Eigendecomposition<double>::Decompose(diag);
    ExpectOrthonormal(result);
}

TEST_F(EigendecompositionTest, EigenvectorOrthogonalityGeneral) {
    PrintSection("Eigenvector Orthogonality (General Symmetric)",
                 "Even with nontrivial off-diagonal entries, the Jacobi "
                 "method must produce orthonormal eigenvectors.");

    std::vector<std::vector<double>> matrix = {
        { 4.0, -2.0,  2.0},
        {-2.0,  2.0, -4.0},
        { 2.0, -4.0, 11.0}
    };

    auto result = Eigendecomposition<double>::Decompose(matrix);
    ExpectOrthonormal(result);
}

// ---------------------------------------------------------------------------
// Rejects non-symmetric input
// ---------------------------------------------------------------------------

TEST_F(EigendecompositionTest, ThrowsOnNonSymmetricMatrix) {
    PrintSection("Non-Symmetric Matrix Rejection",
                 "The Jacobi method is only valid for symmetric matrices. "
                 "The algorithm must throw std::invalid_argument for "
                 "non-symmetric input.");

    std::vector<std::vector<double>> non_symmetric = {
        {1.0, 2.0},
        {3.0, 4.0}  // M[0][1] != M[1][0]
    };

    EXPECT_THROW(Eigendecomposition<double>::Decompose(non_symmetric),
                 std::invalid_argument);
}

TEST_F(EigendecompositionTest, ThrowsOnNonSquareMatrix) {
    PrintSection("Non-Square Matrix Rejection",
                 "A non-square matrix cannot be symmetric and must be "
                 "rejected before attempting decomposition.");

    std::vector<std::vector<double>> non_square = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    EXPECT_THROW(Eigendecomposition<double>::Decompose(non_square),
                 std::invalid_argument);
}

TEST_F(EigendecompositionTest, ThrowsOnEmptyMatrix) {
    PrintSection("Empty Matrix Rejection",
                 "An empty matrix has no eigenvalues and must be rejected.");

    std::vector<std::vector<double>> empty = {};

    EXPECT_THROW(Eigendecomposition<double>::Decompose(empty),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Float-precision template instantiation
// ---------------------------------------------------------------------------

TEST_F(EigendecompositionTest, FloatPrecisionDecomposition) {
    PrintSection("Float Precision Template",
                 "Verify the algorithm works correctly when instantiated "
                 "with float instead of double, exercising the "
                 "std::floating_point concept constraint.");

    std::vector<std::vector<float>> matrix = {
        {4.0f, 1.0f},
        {1.0f, 3.0f}
    };

    auto result = Eigendecomposition<float>::Decompose(matrix);

    // Eigenvalues of [[4,1],[1,3]] are (7+sqrt(5))/2 and (7-sqrt(5))/2.
    std::vector<float> sorted = result.Eigenvalues;
    std::sort(sorted.begin(), sorted.end());

    float expected_low  = (7.0f - std::sqrt(5.0f)) / 2.0f;
    float expected_high = (7.0f + std::sqrt(5.0f)) / 2.0f;

    EXPECT_NEAR(sorted[0], expected_low,  1e-5f);
    EXPECT_NEAR(sorted[1], expected_high, 1e-5f);
}
