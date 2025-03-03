/**
 * @file main.cpp
 * @brief Educational driver for QR-algorithm eigendecomposition of general matrices.
 *
 * Shows the input matrix, decomposes it, verifies A*v = lambda*v,
 * and explains the algorithm steps, Householder reflections, shift
 * acceleration, and complexity.
 */

#include "EigendecompositionGeneral.hpp"

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

void VerifyEigenpairs(const std::string& label,
                      const std::vector<std::vector<double>>& a,
                      const EigendecompositionGeneral<double>::GeneralEigenResult& result) {
    std::size_t n = a.size();
    std::cout << "\n  Verification: A * v = lambda * v  (" << label << ")\n";
    std::cout << "    " << std::setw(10) << std::left << "lambda"
              << std::setw(20) << "max|A*v - lambda*v|"
              << "status\n";
    std::cout << "    " << std::string(40, '-') << "\n";

    for (std::size_t k = 0; k < n; ++k) {
        double lambda = result.Eigenvalues[k];

        // Extract eigenvector column k.
        std::vector<double> v(n);
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = result.Eigenvectors[i][k];
        }

        // Compute A * v.
        std::vector<double> av(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                av[i] += a[i][j] * v[j];
            }
        }

        double max_err = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            max_err = std::max(max_err, std::fabs(av[i] - lambda * v[i]));
        }
        std::cout << "    " << std::setw(10) << std::fixed
                  << std::setprecision(4) << std::right << lambda
                  << std::setw(20) << std::scientific
                  << std::setprecision(2) << max_err
                  << "  " << (max_err < 1e-6 ? "PASS" : "FAIL") << "\n";
    }
}

} // namespace

int main() {
    try {
        std::cout
            << "=============================================\n"
            << "  Eigendecomposition — General Matrices\n"
            << "  (QR Algorithm with Householder + Wilkinson Shift)\n"
            << "=============================================\n\n"

            << "PURPOSE\n"
            << "  Factor a square matrix A into:\n\n"
            << "    A = V * Lambda * V^{-1}\n\n"
            << "  Lambda = diagonal matrix of eigenvalues\n"
            << "  V      = matrix whose columns are eigenvectors\n\n"
            << "  Unlike the symmetric case, V is generally NOT\n"
            << "  orthogonal and eigenvalues can be complex.\n"
            << "  This implementation handles real eigenvalues.\n\n"
            << "  Applications:\n"
            << "    - Stability analysis of dynamical systems (dx/dt = Ax)\n"
            << "    - Markov chain steady-state (dominant eigenvector)\n"
            << "    - Population dynamics (Leslie matrices)\n\n"

            << "HOW IT WORKS — QR Algorithm\n\n"
            << "  Each iteration transforms A into a more upper-triangular\n"
            << "  matrix with the same eigenvalues:\n\n"
            << "    1. DEFLATE: scan sub-diagonal from bottom for converged\n"
            << "       eigenvalues. Shrink the active block.\n"
            << "       Why: avoids wasting iterations on already-converged\n"
            << "       eigenvalues. Critical for non-symmetric matrices.\n\n"
            << "    2. WILKINSON SHIFT: choose sigma from the eigenvalue of\n"
            << "       the bottom 2x2 sub-block closest to a[p-1][p-1].\n"
            << "       Why: cubic convergence for symmetric, quadratic for\n"
            << "       general. Never oscillates, unlike simple Rayleigh.\n\n"
            << "    3. QR DECOMPOSE: A - sigma*I = Q*R (Householder)\n"
            << "       Why: Householder reflections are backward-stable.\n"
            << "       Q is orthogonal to machine precision, unlike\n"
            << "       Gram-Schmidt which loses orthogonality.\n\n"
            << "    4. REVERSE MULTIPLY: A' = R*Q + sigma*I\n"
            << "       This is a similarity transform: A' = Q^T * A * Q.\n"
            << "       Why: similarity preserves eigenvalues while\n"
            << "       pushing A toward upper-triangular form.\n\n"
            << "    5. ACCUMULATE V = V * Q (builds eigenvector matrix)\n\n"
            << "    6. REPEAT until sub-diagonal elements < epsilon\n"
            << "       Diagonal converges to eigenvalues.\n\n"
            << "    7. BACK-SUBSTITUTE: solve (T - lambda*I)x = 0 for each\n"
            << "       eigenvalue in the upper-triangular Schur form, then\n"
            << "       transform: v = V * x.  (Like LAPACK's dtrevc.)\n"
            << "       Why: accumulated Q gives Schur vectors, not true\n"
            << "       eigenvectors for non-symmetric matrices.\n\n"

            << "  HOUSEHOLDER vs GRAM-SCHMIDT QR\n"
            << "    Gram-Schmidt:  Q loses orthogonality in O(eps * kappa(A))\n"
            << "    Householder:   backward-stable, ||Q*R - A|| = O(eps*||A||)\n"
            << "    Both O(n^3), but Householder is the industry standard.\n"
            << "---------------------------------------------\n\n";

        // =============================================================
        // Demo 1: 3x3 non-symmetric matrix
        // =============================================================

        std::cout << "  DEMO 1: 3x3 Non-Symmetric Matrix\n"
                  << "  ---------------------------------\n"
                  << "  Non-symmetric matrices arise in systems where\n"
                  << "  interactions are asymmetric (predator-prey,\n"
                  << "  directed graphs, fluid dynamics).\n"
                  << "  This matrix is constructed as P*D*P^{-1} with\n"
                  << "  D = diag(1, 3, 9), guaranteeing real eigenvalues.\n";

        std::vector<std::vector<double>> a1 = {
            { 5.0, -4.0,  4.0},
            {-1.0,  2.0,  1.0},
            { 3.0, -3.0,  6.0}
        };

        std::cout << "\n  Matrix being decomposed:\n";
        PrintMatrix("A", a1);

        auto result1 = EigendecompositionGeneral<double>::Decompose(a1);

        std::cout << "\n  Eigenvalues:\n";
        for (std::size_t i = 0; i < result1.Eigenvalues.size(); ++i) {
            std::cout << "    lambda_" << i << " = "
                      << std::fixed << std::setprecision(6)
                      << result1.Eigenvalues[i] << "\n";
        }

        double eigenvalue_sum = 0.0;
        double trace = 0.0;
        for (std::size_t i = 0; i < a1.size(); ++i) {
            eigenvalue_sum += result1.Eigenvalues[i];
            trace += a1[i][i];
        }
        std::cout << "\n  Trace check (sum of eigenvalues = trace of A):\n"
                  << "    trace(A)        = " << std::fixed
                  << std::setprecision(4) << trace << "\n"
                  << "    sum(eigenvalues) = " << eigenvalue_sum << "\n";

        std::cout << "\n  Eigenvectors (each column is an eigenvector):\n";
        PrintMatrix("V", result1.Eigenvectors, 6);

        VerifyEigenpairs("3x3 non-symmetric", a1, result1);

        // =============================================================
        // Demo 2: QR decomposition step-by-step
        // =============================================================

        std::cout << "\n\n  DEMO 2: QR Decomposition Internals\n"
                  << "  -----------------------------------\n"
                  << "  Showing Q and R for the same matrix.\n"
                  << "  A = Q * R,  Q orthogonal, R upper triangular.\n";

        auto [q, r] = EigendecompositionGeneral<double>::QrDecomposition(a1);

        std::cout << "\n";
        PrintMatrix("Q (orthogonal)", q, 6);
        std::cout << "\n";
        PrintMatrix("R (upper triangular)", r, 6);

        // Verify A = Q*R
        auto qr = EigendecompositionGeneral<double>::MultiplyMatrices(q, r);
        std::cout << "\n";
        PrintMatrix("Q * R (should match A)", qr, 6);

        // =============================================================
        // Demo 3: symmetric matrix (QR algorithm works for symmetric too)
        // =============================================================

        std::cout << "\n\n  DEMO 3: Symmetric Matrix\n"
                  << "  -----------------------\n"
                  << "  The QR algorithm also works for symmetric matrices.\n"
                  << "  (Jacobi is preferred for symmetric, but QR is valid.)\n"
                  << "  Known eigenvalues: 5, 2, 2.\n";

        std::vector<std::vector<double>> a3 = {
            {3.0, 1.0, 1.0},
            {1.0, 3.0, 1.0},
            {1.0, 1.0, 3.0}
        };

        std::cout << "\n  Matrix being decomposed:\n";
        PrintMatrix("A", a3);

        auto result3 = EigendecompositionGeneral<double>::Decompose(a3);

        std::cout << "\n  Eigenvalues:\n";
        for (std::size_t i = 0; i < result3.Eigenvalues.size(); ++i) {
            std::cout << "    lambda_" << i << " = "
                      << std::fixed << std::setprecision(6)
                      << result3.Eigenvalues[i] << "\n";
        }

        VerifyEigenpairs("3x3 symmetric", a3, result3);

        // =============================================================
        // Demo 4: diagonal matrix (already in Schur form)
        // =============================================================

        std::cout << "\n\n  DEMO 4: Diagonal Matrix (Trivial Case)\n"
                  << "  --------------------------------------\n"
                  << "  A diagonal matrix is already in Schur form.\n"
                  << "  QR should converge immediately.\n";

        std::vector<std::vector<double>> a4 = {
            {5.0, 0.0, 0.0},
            {0.0, 3.0, 0.0},
            {0.0, 0.0, 1.0}
        };

        std::cout << "\n  Matrix being decomposed:\n";
        PrintMatrix("A", a4);

        auto result4 = EigendecompositionGeneral<double>::Decompose(a4);

        std::cout << "\n  Eigenvalues:\n";
        for (std::size_t i = 0; i < result4.Eigenvalues.size(); ++i) {
            std::cout << "    lambda_" << i << " = "
                      << std::fixed << std::setprecision(6)
                      << result4.Eigenvalues[i] << "\n";
        }

        // =============================================================
        // Complexity summary
        // =============================================================

        std::cout << "\n\n---------------------------------------------\n"
                  << "  TIME AND SPACE COMPLEXITY\n"
                  << "---------------------------------------------\n\n"
                  << "  This implementation (Householder QR + Wilkinson shift):\n"
                  << "    Time:  O(n^3) per QR step * O(n) steps = O(n^4)\n"
                  << "    Space: O(n^2)\n\n"
                  << "  LAPACK dgeev (Hessenberg + implicit double-shift QR):\n"
                  << "    Time:  O(n^3)   — each QR step is O(n^2) on Hessenberg\n"
                  << "    Space: O(n^2)\n\n"
                  << "  How to make this faster:\n"
                  << "    1. Reduce to upper Hessenberg form first (Householder,\n"
                  << "       O(10n^3/3)), then each QR step costs O(n^2).\n"
                  << "    2. Use implicit double-shift (Francis) QR for complex\n"
                  << "       eigenvalue pairs without complex arithmetic.\n"
                  << "    3. Deflation (already implemented) shrinks the active\n"
                  << "       block as eigenvalues converge.\n\n"
                  << "  Per-function breakdown:\n"
                  << "    Decompose:              O(n^4) time, O(n^2) space\n"
                  << "    QrDecomposition:        O(n^3) time, O(n^2) space\n"
                  << "    MultiplyMatrices:       O(n^3) time, O(n^2) space\n"
                  << "    OffDiagonalFrobeniusNorm: O(n^2) time, O(1) space\n"
                  << "    DotProduct / VectorNorm: O(n) time, O(1) space\n";

        std::cout << "\nDone.\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
