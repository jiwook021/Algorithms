/**
 * @file test_SingularValueDecomposition.cpp
 * @brief Educational unit tests for SVD via deflated power iteration.
 *
 * Each test prints a section banner explaining *what* is being tested and
 * *why*, followed by [Result] lines showing computed vs expected values
 * before the assertion.  This makes test output readable as a tutorial.
 *
 * Coverage:
 *   - Utility functions (transpose, multiply, dot product)
 *   - Decomposition reconstruction (A ~= U * Sigma * Vt)
 *   - Orthogonality of U columns
 *   - Singular values in descending order
 *   - Condition number
 *   - Rank for full-rank and rank-deficient matrices
 *   - Frobenius norm
 *   - Edge cases (empty matrix, float instantiation)
 */

#include "SingularValueDecomposition.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: educational banner printed at the start of every test
// ---------------------------------------------------------------------------

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

} // namespace

// ---------------------------------------------------------------------------
// Type aliases for the default double instantiation
// ---------------------------------------------------------------------------

using SVD    = SingularValueDecomposition<double>;
using Matrix = SVD::Matrix;
using Vector = SVD::Vector;

static constexpr double kTol = 1e-6;

static bool Near(double a, double b, double eps = kTol) {
    return std::fabs(a - b) < eps;
}

// ======================== Utility Tests ====================================

void TestTranspose() {
    PrintSection("Transpose",
                 "Transpose is a fundamental building block used in "
                 "reconstruction (A = U * Sigma * Vt) and pseudoinverse.");

    Matrix a = {{1, 2, 3}, {4, 5, 6}};
    auto t = SVD::Transpose(a);

    std::cout << "[Result] Transpose shape: " << t.size() << " x "
              << t[0].size() << "  (expected 3 x 2)" << std::endl;
    std::cout << "[Result] T[0][1] = " << t[0][1]
              << "  (expected 4)" << std::endl;
    std::cout << "[Result] T[2][0] = " << t[2][0]
              << "  (expected 3)" << std::endl;

    assert(t.size() == 3 && t[0].size() == 2);
    assert(Near(t[0][1], 4.0));
    assert(Near(t[2][0], 3.0));
    std::cout << "  PASSED\n";
}

void TestMatrixMultiply() {
    PrintSection("Matrix Multiply",
                 "Matrix multiplication is used to reconstruct A from "
                 "U * Sigma * Vt. Verifying with identity preserves B.");

    Matrix id = {{1, 0}, {0, 1}};
    Matrix b  = {{5, 6}, {7, 8}};
    auto c = SVD::MatrixMultiply(id, b);

    std::cout << "[Result] (I * B)[0][0] = " << c[0][0]
              << "  (expected 5)" << std::endl;
    std::cout << "[Result] (I * B)[1][1] = " << c[1][1]
              << "  (expected 8)" << std::endl;

    assert(Near(c[0][0], 5.0) && Near(c[1][1], 8.0));
    std::cout << "  PASSED\n";
}

void TestDotProduct() {
    PrintSection("Dot Product",
                 "Dot product is used internally for norm computation "
                 "during power iteration. <[1,2,3],[4,5,6]> = 32.");

    double result = SVD::DotProduct({1, 2, 3}, {4, 5, 6});

    std::cout << "[Result] DotProduct = " << result
              << "  (expected 32)" << std::endl;

    assert(Near(result, 32.0));
    std::cout << "  PASSED\n";
}

// ======================== Core SVD Tests ===================================

void TestReconstruction2x3() {
    PrintSection("Reconstruction (2x3)",
                 "The fundamental SVD identity: A = U * Sigma * Vt. "
                 "If reconstruction error is near zero, the decomposition "
                 "captured all the information in A.");

    Matrix a = {{3.0, 2.0, 2.0}, {2.0, 3.0, -2.0}};
    auto result = SVD::Decompose(a);
    auto rec    = SVD::Reconstruct(result);

    double max_error = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a[0].size(); ++j) {
            double err = std::fabs(rec[i][j] - a[i][j]);
            if (err > max_error) max_error = err;
            std::cout << "[Result] A[" << i << "][" << j << "] = "
                      << a[i][j] << "  reconstructed = "
                      << std::setprecision(10) << rec[i][j] << std::endl;
        }
    }
    std::cout << "[Result] Max element-wise error = "
              << max_error << "  (threshold 0.5)" << std::endl;

    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a[0].size(); ++j) {
            assert(Near(rec[i][j], a[i][j], 0.5));
        }
    }
    std::cout << "  PASSED\n";
}

void TestSquareReconstruction() {
    PrintSection("Reconstruction (2x2 square)",
                 "Square matrices are the most common case. "
                 "Verifying A = U * Sigma * Vt for a non-symmetric matrix.");

    Matrix a = {{4, 0}, {3, -5}};
    auto result = SVD::Decompose(a);
    auto rec    = SVD::Reconstruct(result);

    double max_error = 0.0;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            double err = std::fabs(rec[i][j] - a[i][j]);
            if (err > max_error) max_error = err;
            std::cout << "[Result] A[" << i << "][" << j << "] = "
                      << a[i][j] << "  reconstructed = "
                      << std::setprecision(10) << rec[i][j] << std::endl;
        }
    }
    std::cout << "[Result] Max element-wise error = "
              << max_error << "  (threshold 0.5)" << std::endl;

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            assert(Near(rec[i][j], a[i][j], 0.5));
        }
    }
    std::cout << "  PASSED\n";
}

void TestOrthogonalityOfU() {
    PrintSection("Orthogonality of U",
                 "Left singular vectors must be orthonormal: U^T * U = I. "
                 "This guarantees the decomposition preserves angles and "
                 "lengths in the column space.");

    Matrix a = {{3.0, 2.0, 2.0}, {2.0, 3.0, -2.0}};
    auto result = SVD::Decompose(a);
    const auto& u = result.U;
    std::size_t m = u.size();

    auto ut  = SVD::Transpose(u);
    auto utu = SVD::MatrixMultiply(ut, u);

    double max_off_diag = 0.0;
    double max_diag_err = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            double err = std::fabs(utu[i][j] - expected);
            if (i == j) {
                if (err > max_diag_err) max_diag_err = err;
            } else {
                if (err > max_off_diag) max_off_diag = err;
            }
        }
    }

    std::cout << "[Result] Max diagonal deviation from 1.0  = "
              << max_diag_err << std::endl;
    std::cout << "[Result] Max off-diagonal deviation from 0 = "
              << max_off_diag << std::endl;

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            assert(Near(utu[i][j], expected, 1e-4));
        }
    }
    std::cout << "  PASSED\n";
}

void TestSingularValuesDescending() {
    PrintSection("Singular Values Descending",
                 "SVD convention orders singular values s_1 >= s_2 >= ... >= 0. "
                 "This ordering is essential for low-rank approximation "
                 "(keep the top-k values).");

    Matrix a = {{3.0, 2.0, 2.0}, {2.0, 3.0, -2.0}};
    auto result = SVD::Decompose(a);

    std::cout << "[Result] Singular values:";
    for (auto sv : result.SingularValues) {
        std::cout << "  " << sv;
    }
    std::cout << std::endl;

    for (std::size_t i = 1; i < result.SingularValues.size(); ++i) {
        std::cout << "[Result] s[" << i - 1 << "]=" << result.SingularValues[i - 1]
                  << " >= s[" << i << "]=" << result.SingularValues[i]
                  << "  => " << (result.SingularValues[i - 1] >=
                                 result.SingularValues[i] - kTol
                                     ? "OK" : "FAIL")
                  << std::endl;
        assert(result.SingularValues[i - 1] >=
               result.SingularValues[i] - kTol);
    }
    std::cout << "  PASSED\n";
}

void TestConditionNumberIdentity() {
    PrintSection("Condition Number (Identity)",
                 "The condition number kappa(A) = sigma_max / sigma_min. "
                 "For the identity matrix all singular values are 1, "
                 "so kappa = 1. A large kappa signals numerical instability.");

    Matrix a = {{1, 0}, {0, 1}};
    auto result = SVD::Decompose(a);
    double cond  = SVD::ConditionNumber(result);

    std::cout << "[Result] Condition number = " << cond
              << "  (expected 1.0)" << std::endl;
    std::cout << "[Result] Singular values:";
    for (auto sv : result.SingularValues) {
        std::cout << "  " << sv;
    }
    std::cout << "  (expected [1, 1])" << std::endl;

    assert(Near(cond, 1.0));
    std::cout << "  PASSED\n";
}

void TestConditionNumberIllConditioned() {
    PrintSection("Condition Number (Ill-conditioned)",
                 "A matrix with widely separated singular values has a "
                 "large condition number, indicating sensitivity to "
                 "perturbations in the input.");

    // Diagonal matrix with sigma = 100 and sigma = 0.01.
    Matrix a = {{100, 0}, {0, 0.01}};
    auto result = SVD::Decompose(a);
    double cond  = SVD::ConditionNumber(result);

    std::cout << "[Result] Condition number = " << cond
              << "  (expected ~10000)" << std::endl;

    assert(Near(cond, 10000.0, 1.0));
    std::cout << "  PASSED\n";
}

void TestRankFullRank() {
    PrintSection("Rank (Full-rank matrix)",
                 "The numerical rank counts singular values above a tolerance. "
                 "The 2x2 identity has rank 2.");

    Matrix a = {{1, 0}, {0, 1}};
    auto result = SVD::Decompose(a);
    auto r = SVD::Rank(result);

    std::cout << "[Result] Rank = " << r
              << "  (expected 2)" << std::endl;

    assert(r == 2);
    std::cout << "  PASSED\n";
}

void TestRankDeficient() {
    PrintSection("Rank (Rank-deficient matrix)",
                 "The matrix [[1,2],[2,4]] has linearly dependent rows, "
                 "so its rank should be 1 (only one nonzero singular value).");

    Matrix a = {{1, 2}, {2, 4}};
    auto result = SVD::Decompose(a);
    auto r = SVD::Rank(result);

    std::cout << "[Result] Rank = " << r
              << "  (expected 1)" << std::endl;
    std::cout << "[Result] Singular values:";
    for (auto sv : result.SingularValues) {
        std::cout << "  " << sv;
    }
    std::cout << std::endl;

    assert(r == 1);
    std::cout << "  PASSED\n";
}

void TestFrobeniusNorm() {
    PrintSection("Frobenius Norm",
                 "The Frobenius norm equals sqrt(sum of squared singular "
                 "values). For the 2x2 identity: sqrt(1^2 + 1^2) = sqrt(2).");

    Matrix a = {{1, 0}, {0, 1}};
    auto result = SVD::Decompose(a);
    double norm  = SVD::FrobeniusNorm(result);

    std::cout << "[Result] Frobenius norm = " << norm
              << "  (expected " << std::sqrt(2.0) << ")" << std::endl;

    assert(Near(norm, std::sqrt(2.0)));
    std::cout << "  PASSED\n";
}

void TestEmptyThrows() {
    PrintSection("Empty Matrix Throws",
                 "An empty input is invalid; the implementation must "
                 "reject it with std::invalid_argument.");

    bool threw = false;
    try {
        SVD::Decompose({});
    } catch (const std::invalid_argument& e) {
        threw = true;
        std::cout << "[Result] Caught exception: " << e.what() << std::endl;
    }

    std::cout << "[Result] Exception thrown = "
              << (threw ? "true" : "false")
              << "  (expected true)" << std::endl;

    assert(threw);
    std::cout << "  PASSED\n";
}

// ======================== Float Instantiation Test =========================

void TestFloatInstantiation() {
    PrintSection("Float Instantiation",
                 "The template is constrained by std::floating_point. "
                 "Verifying that SingularValueDecomposition<float> compiles "
                 "and produces a valid decomposition.");

    using SVDf = SingularValueDecomposition<float>;
    SVDf::Matrix a = {{3.0f, 2.0f}, {2.0f, 3.0f}};
    auto result = SVDf::Decompose(a);
    auto rec    = SVDf::Reconstruct(result);

    float max_error = 0.0f;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            float err = std::fabs(rec[i][j] - a[i][j]);
            if (err > max_error) max_error = err;
        }
    }

    std::cout << "[Result] float max reconstruction error = "
              << max_error << "  (threshold 0.5)" << std::endl;
    std::cout << "[Result] Singular values:";
    for (auto sv : result.SingularValues) {
        std::cout << "  " << sv;
    }
    std::cout << std::endl;

    assert(max_error < 0.5f);
    std::cout << "  PASSED\n";
}

// ======================== Main =============================================

int main() {
    std::cout << "============================================\n";
    std::cout << "  Singular Value Decomposition -- Tests\n";
    std::cout << "============================================\n";

    // Utilities
    TestTranspose();
    TestMatrixMultiply();
    TestDotProduct();

    // Core decomposition
    TestReconstruction2x3();
    TestSquareReconstruction();
    TestOrthogonalityOfU();
    TestSingularValuesDescending();

    // Derived quantities
    TestConditionNumberIdentity();
    TestConditionNumberIllConditioned();
    TestRankFullRank();
    TestRankDeficient();
    TestFrobeniusNorm();

    // Edge cases
    TestEmptyThrows();

    // Template instantiation
    TestFloatInstantiation();

    std::cout << "\n============================================\n";
    std::cout << "  All SingularValueDecomposition tests PASSED\n";
    std::cout << "============================================\n";

    return EXIT_SUCCESS;
}
