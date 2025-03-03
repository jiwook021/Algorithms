/**
 * @file main.cpp
 * @brief Educational driver for Jacobi eigendecomposition of symmetric matrices.
 *
 * Shows the input matrix, decomposes it, verifies A*v = lambda*v,
 * and explains the algorithm, complexity, and real-world context.
 */

#include "Eigendecomposition.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void PrintMatrix(const std::string& label,
                 const std::vector<std::vector<double>>& m,
                 int precision = 4) {
    std::cout << "  " << label << ":\n";
    for (const auto& row : m) {
        std::cout << "    [";
        for (std::size_t j = 0; j < row.size(); ++j) {
            std::cout << std::setw(9) << std::fixed
                      << std::setprecision(precision) << row[j];
        }
        std::cout << " ]\n";
    }
}

std::vector<double> MatVec(const std::vector<std::vector<double>>& a,
                           const std::vector<double>& v) {
    std::size_t n = a.size();
    std::vector<double> result(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[i] += a[i][j] * v[j];
        }
    }
    return result;
}

void VerifyEigenpairs(const std::string& label,
                      const std::vector<std::vector<double>>& a,
                      const Eigendecomposition<double>::EigenResult& result) {
    std::size_t n = a.size();
    std::cout << "\n  Verification: A * v = lambda * v  (" << label << ")\n";
    std::cout << "    " << std::setw(10) << std::left << "lambda"
              << std::setw(20) << "max|A*v - lambda*v|"
              << "status\n";
    std::cout << "    " << std::string(40, '-') << "\n";

    for (std::size_t k = 0; k < n; ++k) {
        double lambda = result.Eigenvalues[k];
        std::vector<double> v(n);
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = result.Eigenvectors[i][k];
        }
        auto av = MatVec(a, v);
        double max_err = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            max_err = std::max(max_err, std::fabs(av[i] - lambda * v[i]));
        }
        std::cout << "    " << std::setw(10) << std::fixed
                  << std::setprecision(4) << std::right << lambda
                  << std::setw(20) << std::scientific
                  << std::setprecision(2) << max_err
                  << "  " << (max_err < 1e-8 ? "PASS" : "FAIL") << "\n";
    }
}

} // namespace

int main() {
    try {
        std::cout
            << "=============================================\n"
            << "  Eigendecomposition — Symmetric Matrices\n"
            << "  (Jacobi Rotation Method)\n"
            << "=============================================\n\n"

            << "PURPOSE\n"
            << "  Factor a real symmetric matrix A into:\n\n"
            << "    A = V * Lambda * V^T\n\n"
            << "  Lambda = diagonal matrix of eigenvalues\n"
            << "  V      = orthogonal eigenvector matrix (V^T V = I)\n\n"
            << "  The Spectral Theorem guarantees every real symmetric\n"
            << "  matrix has this factorization with real eigenvalues\n"
            << "  and orthogonal eigenvectors.\n\n"
            << "  Applications:\n"
            << "    - PCA: directions of maximum variance\n"
            << "    - Vibration analysis: natural frequencies / mode shapes\n"
            << "    - Quantum mechanics: energy levels / wavefunctions\n\n"

            << "HOW IT WORKS — Jacobi Rotation Method\n\n"
            << "  Each iteration:\n"
            << "    1. FIND the largest off-diagonal |a[p][q]|\n"
            << "       Why: zeroing the biggest gives fastest convergence.\n\n"
            << "    2. COMPUTE angle theta so that a'[p][q] = 0:\n"
            << "       tan(2*theta) = 2*a[p][q] / (a[q][q] - a[p][p])\n"
            << "       Why: this angle makes the (p,q) entry exactly zero\n"
            << "       after the rotation.\n\n"
            << "    3. APPLY similarity transform: A' = G^T * A * G\n"
            << "       G is a Givens rotation in the (p,q) plane.\n"
            << "       Why: similarity transforms preserve eigenvalues.\n"
            << "       The (p,q) entry is now zero, but others shift.\n\n"
            << "    4. ACCUMULATE V = V * G (builds eigenvector matrix)\n"
            << "       Why: V converges to the matrix whose columns are\n"
            << "       the eigenvectors of the original A.\n\n"
            << "    5. REPEAT until all off-diagonal |a[i][j]| < epsilon\n"
            << "       Diagonal converges to eigenvalues.\n\n"

            << "  WHY JACOBI FOR SYMMETRIC MATRICES?\n"
            << "    - Givens rotations preserve symmetry at every step.\n"
            << "    - Each rotation strictly reduces off-diagonal energy.\n"
            << "    - Guaranteed convergence to real eigenvalues.\n"
            << "    - Inherently parallelizable (independent rotations).\n"
            << "---------------------------------------------\n\n";

        // =============================================================
        // Demo 1: 3x3 symmetric matrix
        // =============================================================

        std::cout << "  DEMO 1: 3x3 Symmetric Matrix\n"
                  << "  ----------------------------\n";

        std::vector<std::vector<double>> a1 = {
            { 4.0, -2.0,  2.0},
            {-2.0,  2.0, -4.0},
            { 2.0, -4.0, 11.0}
        };

        std::cout << "\n  Matrix being decomposed:\n";
        PrintMatrix("A", a1);

        auto result1 = Eigendecomposition<double>::Decompose(a1);

        std::cout << "\n  Eigenvalues:\n";
        for (std::size_t i = 0; i < result1.Eigenvalues.size(); ++i) {
            std::cout << "    lambda_" << i << " = "
                      << std::fixed << std::setprecision(6)
                      << result1.Eigenvalues[i] << "\n";
        }

        std::cout << "\n  Eigenvectors (each column is an eigenvector):\n";
        PrintMatrix("V", result1.Eigenvectors, 6);

        VerifyEigenpairs("3x3", a1, result1);

        // Check orthogonality
        std::size_t n1 = a1.size();
        std::cout << "\n  Orthogonality check: V^T * V should be I\n";
        for (std::size_t i = 0; i < n1; ++i) {
            for (std::size_t j = i; j < n1; ++j) {
                double dot = 0.0;
                for (std::size_t k = 0; k < n1; ++k) {
                    dot += result1.Eigenvectors[k][i]
                         * result1.Eigenvectors[k][j];
                }
                double expected = (i == j) ? 1.0 : 0.0;
                std::cout << "    v_" << i << " . v_" << j << " = "
                          << std::fixed << std::setprecision(8) << dot
                          << "  (expected " << expected << ")\n";
            }
        }

        // =============================================================
        // Demo 2: Diagonal matrix (trivial — eigenvalues are diagonal)
        // =============================================================

        std::cout << "\n\n  DEMO 2: Diagonal Matrix (Trivial Case)\n"
                  << "  --------------------------------------\n"
                  << "  A diagonal matrix is already decomposed:\n"
                  << "  eigenvalues = diagonal entries, eigenvectors = I.\n"
                  << "  Jacobi should converge immediately (0 rotations).\n";

        std::vector<std::vector<double>> a2 = {
            {7.0, 0.0, 0.0},
            {0.0, 3.0, 0.0},
            {0.0, 0.0, 5.0}
        };

        std::cout << "\n  Matrix being decomposed:\n";
        PrintMatrix("A", a2);

        auto result2 = Eigendecomposition<double>::Decompose(a2);

        std::cout << "\n  Eigenvalues:\n";
        for (std::size_t i = 0; i < result2.Eigenvalues.size(); ++i) {
            std::cout << "    lambda_" << i << " = "
                      << std::fixed << std::setprecision(6)
                      << result2.Eigenvalues[i] << "\n";
        }

        // =============================================================
        // Demo 3: 4x4 symmetric — PCA-like covariance matrix
        // =============================================================

        std::cout << "\n\n  DEMO 3: 4x4 Symmetric Matrix\n"
                  << "  ----------------------------\n"
                  << "  A covariance-like matrix.  In PCA, the largest\n"
                  << "  eigenvalue's eigenvector is the first principal\n"
                  << "  component (direction of maximum variance).\n";

        std::vector<std::vector<double>> a3 = {
            { 5.0,  2.0, -1.0,  0.0},
            { 2.0,  6.0,  0.0,  1.0},
            {-1.0,  0.0,  4.0, -2.0},
            { 0.0,  1.0, -2.0,  3.0}
        };

        std::cout << "\n  Matrix being decomposed:\n";
        PrintMatrix("A", a3);

        auto result3 = Eigendecomposition<double>::Decompose(a3);

        std::cout << "\n  Eigenvalues:\n";
        for (std::size_t i = 0; i < result3.Eigenvalues.size(); ++i) {
            std::cout << "    lambda_" << i << " = "
                      << std::fixed << std::setprecision(6)
                      << result3.Eigenvalues[i] << "\n";
        }

        double eigenvalue_sum = 0.0;
        double trace = 0.0;
        for (std::size_t i = 0; i < a3.size(); ++i) {
            eigenvalue_sum += result3.Eigenvalues[i];
            trace += a3[i][i];
        }
        std::cout << "\n  Trace of A:          " << std::fixed
                  << std::setprecision(4) << trace
                  << "\n  Sum of eigenvalues:  " << eigenvalue_sum
                  << "  (must match trace)\n";

        VerifyEigenpairs("4x4", a3, result3);

        // =============================================================
        // Complexity summary
        // =============================================================

        std::cout << "\n\n---------------------------------------------\n"
                  << "  TIME AND SPACE COMPLEXITY\n"
                  << "---------------------------------------------\n\n"
                  << "  Jacobi Method (this implementation):\n"
                  << "    Time:  O(n^3) per sweep * O(n) sweeps = O(n^4)\n"
                  << "    Space: O(n^2)\n\n"
                  << "  LAPACK dsyev (Householder + tridiagonal QR):\n"
                  << "    Time:  O(n^3)   — asymptotically faster\n"
                  << "    Space: O(n^2)\n\n"
                  << "  When to prefer Jacobi:\n"
                  << "    - Small dense matrices (n < 100)\n"
                  << "    - When high relative accuracy is needed\n"
                  << "    - Parallel implementations (GPU, SIMD)\n\n"
                  << "  Per-function breakdown:\n"
                  << "    Decompose:          O(n^4) time, O(n^2) space\n"
                  << "    IsSymmetric:        O(n^2) time, O(1) space\n"
                  << "    CreateIdentityMatrix: O(n^2) time, O(n^2) space\n";

        std::cout << "\nDone.\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
