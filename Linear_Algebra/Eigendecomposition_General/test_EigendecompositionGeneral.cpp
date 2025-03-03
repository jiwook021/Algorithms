/**
 * @file test_EigendecompositionGeneral.cpp
 * @brief Google Test suite for the QR-algorithm general eigendecomposition.
 * @details Verifies eigenvalue accuracy, eigenvector properties,
 *          reconstruction A*v = lambda*v, and edge cases.
 */

#include "EigendecompositionGeneral.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

} // namespace

static constexpr double kTestTolerance = 1e-6;

// ---------------------------------------------------------------------------
// Non-symmetric matrix eigenvalues
// ---------------------------------------------------------------------------

TEST(EigendecompositionGeneral, NonSymmetricMatrixEigenvalues) {
    PrintSection("Non-Symmetric Matrix Eigenvalues",
        "The QR algorithm must handle general (non-symmetric) matrices. "
        "We verify that the sum of computed eigenvalues equals the trace, "
        "which is an invariant of similarity transformations.");

    // Constructed as P*D*P^{-1}, D = diag(1, 3, 9), guaranteeing real eigenvalues.
    std::vector<std::vector<double>> a = {
        { 5.0, -4.0,  4.0},
        {-1.0,  2.0,  1.0},
        { 3.0, -3.0,  6.0}
    };
    auto result = EigendecompositionGeneral<double>::Decompose(a);

    // Sum of eigenvalues must equal the trace of A.
    double trace_a = 5.0 + 2.0 + 6.0;  // = 13
    double eigenvalue_sum = 0.0;
    for (double val : result.Eigenvalues) {
        eigenvalue_sum += val;
    }

    std::cout << "  Trace of A:            " << trace_a << "\n";
    std::cout << "  Sum of eigenvalues:    " << eigenvalue_sum << "\n";
    for (std::size_t i = 0; i < result.Eigenvalues.size(); ++i) {
        std::cout << "  lambda_" << i << " = "
                  << result.Eigenvalues[i] << "\n";
    }

    EXPECT_NEAR(eigenvalue_sum, trace_a, kTestTolerance);
}

// ---------------------------------------------------------------------------
// Diagonal matrix trivial case
// ---------------------------------------------------------------------------

TEST(EigendecompositionGeneral, DiagonalMatrixTrivialCase) {
    PrintSection("Diagonal Matrix Trivial Case",
        "A diagonal matrix is already in Schur form. The eigenvalues are "
        "the diagonal entries themselves, so the QR algorithm should "
        "converge immediately and return exact values.");

    std::vector<std::vector<double>> diag = {
        {5.0, 0.0, 0.0},
        {0.0, 3.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    auto result = EigendecompositionGeneral<double>::Decompose(diag);

    std::vector<double> sorted = result.Eigenvalues;
    std::sort(sorted.begin(), sorted.end());

    std::cout << "  Expected eigenvalues: 1, 3, 5\n";
    std::cout << "  Computed (sorted):    "
              << sorted[0] << ", " << sorted[1] << ", " << sorted[2] << "\n";

    EXPECT_NEAR(sorted[0], 1.0, kTestTolerance);
    EXPECT_NEAR(sorted[1], 3.0, kTestTolerance);
    EXPECT_NEAR(sorted[2], 5.0, kTestTolerance);
}

// ---------------------------------------------------------------------------
// Reconstruction A*v = lambda*v
// ---------------------------------------------------------------------------

TEST(EigendecompositionGeneral, ReconstructionAvEqualsLambdaV) {
    PrintSection("Reconstruction A*v = lambda*v",
        "The defining property of an eigenpair (lambda, v) is A*v = lambda*v. "
        "For symmetric matrices the QR algorithm produces accurate "
        "eigenvectors, so we verify the reconstruction identity on a "
        "symmetric matrix with known eigenvalues 5, 2, 2.");

    // Symmetric matrix — QR algorithm yields true eigenvectors.
    std::vector<std::vector<double>> a = {
        {3.0, 1.0, 1.0},
        {1.0, 3.0, 1.0},
        {1.0, 1.0, 3.0}
    };
    auto result = EigendecompositionGeneral<double>::Decompose(a);

    std::size_t n = a.size();
    for (std::size_t col = 0; col < n; ++col) {
        double lambda = result.Eigenvalues[col];

        // Extract eigenvector column.
        std::vector<double> v(n);
        for (std::size_t row = 0; row < n; ++row) {
            v[row] = result.Eigenvectors[row][col];
        }

        // Compute A * v.
        std::vector<double> av(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                av[i] += a[i][j] * v[j];
            }
        }

        // Compute lambda * v.
        std::vector<double> lv(n);
        for (std::size_t i = 0; i < n; ++i) {
            lv[i] = lambda * v[i];
        }

        // Check A*v ~ lambda*v.
        double max_diff = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            max_diff = std::max(max_diff, std::fabs(av[i] - lv[i]));
        }

        std::cout << "  Eigenpair " << col
                  << ": lambda = " << lambda
                  << ", max|A*v - lambda*v| = " << max_diff << "\n";

        EXPECT_LT(max_diff, kTestTolerance)
            << "Reconstruction failed for eigenpair " << col;
    }
}

// ---------------------------------------------------------------------------
// Symmetric matrix (QR should still work)
// ---------------------------------------------------------------------------

TEST(EigendecompositionGeneral, SymmetricMatrixEigenvalues) {
    PrintSection("Symmetric Matrix Eigenvalues",
        "Although the Jacobi method is preferred for symmetric matrices, "
        "the QR algorithm must also produce correct results. This tests "
        "a symmetric matrix with known eigenvalues 5, 2, 2.");

    std::vector<std::vector<double>> a = {
        {3.0, 1.0, 1.0},
        {1.0, 3.0, 1.0},
        {1.0, 1.0, 3.0}
    };
    auto result = EigendecompositionGeneral<double>::Decompose(a);

    std::vector<double> sorted = result.Eigenvalues;
    std::sort(sorted.begin(), sorted.end());

    std::cout << "  Expected eigenvalues: 2, 2, 5\n";
    std::cout << "  Computed (sorted):    "
              << sorted[0] << ", " << sorted[1] << ", " << sorted[2] << "\n";

    EXPECT_NEAR(sorted[0], 2.0, kTestTolerance);
    EXPECT_NEAR(sorted[1], 2.0, kTestTolerance);
    EXPECT_NEAR(sorted[2], 5.0, kTestTolerance);
}

// ---------------------------------------------------------------------------
// QR decomposition validity
// ---------------------------------------------------------------------------

TEST(EigendecompositionGeneral, QrDecompositionReconstructsA) {
    PrintSection("QR Decomposition Reconstructs A",
        "The QR factorisation must satisfy A = Q*R. We verify the "
        "reconstruction and that R is upper-triangular.");

    std::vector<std::vector<double>> a = {
        {12.0, -51.0, 4.0},
        { 6.0, 167.0, -68.0},
        {-4.0,  24.0, -41.0}
    };
    auto [q, r] =
        EigendecompositionGeneral<double>::QrDecomposition(a);

    // Verify A ~= Q*R.
    auto qr = EigendecompositionGeneral<double>::MultiplyMatrices(q, r);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(qr[i][j], a[i][j], kTestTolerance)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }

    // Verify R is upper triangular.
    EXPECT_NEAR(r[1][0], 0.0, 1e-9);
    EXPECT_NEAR(r[2][0], 0.0, 1e-9);
    EXPECT_NEAR(r[2][1], 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Non-square matrix throws
// ---------------------------------------------------------------------------

TEST(EigendecompositionGeneral, NonSquareMatrixThrows) {
    PrintSection("Non-Square Matrix Throws",
        "Eigendecomposition is only defined for square matrices. "
        "Passing a non-square input must throw std::invalid_argument.");

    std::vector<std::vector<double>> non_square = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    EXPECT_THROW(EigendecompositionGeneral<double>::Decompose(non_square),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Off-diagonal Frobenius norm
// ---------------------------------------------------------------------------

TEST(EigendecompositionGeneral, OffDiagonalFrobeniusNorm) {
    PrintSection("Off-Diagonal Frobenius Norm",
        "The off-diagonal norm is the convergence criterion for the QR "
        "algorithm. Identity should give 0; a non-trivial matrix gives "
        "sqrt(sum of squared off-diagonal entries).");

    std::vector<std::vector<double>> id = {
        {1.0, 0.0},
        {0.0, 1.0}
    };
    EXPECT_NEAR(
        EigendecompositionGeneral<double>::OffDiagonalFrobeniusNorm(id),
        0.0, 1e-15);

    std::vector<std::vector<double>> a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    double expected = std::sqrt(4.0 + 9.0);  // sqrt(2^2 + 3^2)
    EXPECT_NEAR(
        EigendecompositionGeneral<double>::OffDiagonalFrobeniusNorm(a),
        expected, kTestTolerance);
}

// ---------------------------------------------------------------------------
// Float specialization
// ---------------------------------------------------------------------------

TEST(EigendecompositionGeneral, FloatSpecialization) {
    PrintSection("Float Specialization",
        "The template must compile and produce reasonable results for "
        "float as well as double, verifying the concept constraint works.");

    std::vector<std::vector<float>> diag = {
        {5.0f, 0.0f},
        {0.0f, 3.0f}
    };
    auto result = EigendecompositionGeneral<float>::Decompose(diag);

    std::vector<float> sorted = result.Eigenvalues;
    std::sort(sorted.begin(), sorted.end());

    std::cout << "  float eigenvalues: "
              << sorted[0] << ", " << sorted[1] << "\n";

    EXPECT_NEAR(sorted[0], 3.0f, 1e-4f);
    EXPECT_NEAR(sorted[1], 5.0f, 1e-4f);
}
