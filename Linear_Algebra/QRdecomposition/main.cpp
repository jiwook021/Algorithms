/**
 * @file main.cpp
 * @brief Interactive driver for QR decomposition.
 *
 * Shows what QR decomposition is, why it matters, walks through
 * a concrete least-squares example, and discusses GPU alternatives.
 */

#include "QrDecomposition.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

static void PrintMatrix(const std::string& label, const Matrix& m) {
    std::cout << label << "\n";
    for (const auto& row : m) {
        std::cout << "    ";
        for (double val : row) {
            std::cout << std::setw(10) << std::fixed
                      << std::setprecision(4) << val << " ";
        }
        std::cout << "\n";
    }
}

static void PrintVector(const std::string& label, const Vector& v) {
    std::cout << label << " [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << v[i];
    }
    std::cout << "]\n";
}

/**
 * @brief Multiply two matrices.
 */
static Matrix MultiplyMat(const Matrix& A, const Matrix& B) {
    std::size_t m = A.size(), n = B[0].size(), k = A[0].size();
    Matrix C(m, Vector(n, 0.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t l = 0; l < k; ++l)
                C[i][j] += A[i][l] * B[l][j];
    return C;
}

/**
 * @brief Transpose a matrix.
 */
static Matrix Transpose(const Matrix& M) {
    std::size_t m = M.size(), n = M[0].size();
    Matrix T(n, Vector(m));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            T[j][i] = M[i][j];
    return T;
}

/**
 * @brief Multiply matrix by a vector.
 */
static Vector MatVecMul(const Matrix& M, const Vector& v) {
    Vector result(M.size(), 0.0);
    for (std::size_t i = 0; i < M.size(); ++i)
        for (std::size_t j = 0; j < v.size(); ++j)
            result[i] += M[i][j] * v[j];
    return result;
}

/**
 * @brief Solve upper-triangular system R * x = b by back-substitution.
 *
 *   Start from the last row (one unknown), then work up.
 *   Each row has one more known value than the previous.
 */
static Vector BackSubstitute(const Matrix& R, const Vector& b) {
    std::size_t n = R.size();
    Vector x(n, 0.0);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double sum = b[static_cast<std::size_t>(i)];
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < n; ++j) {
            sum -= R[static_cast<std::size_t>(i)][j]
                   * x[j];
        }
        x[static_cast<std::size_t>(i)] =
            sum / R[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
    }
    return x;
}

/**
 * @brief Compute dot product of column col_a and col_b from matrix M.
 */
static double ColumnDot(const Matrix& M,
                        std::size_t col_a, std::size_t col_b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < M.size(); ++i)
        sum += M[i][col_a] * M[i][col_b];
    return sum;
}

int main() {
    std::cout
        << "=============================================\n"
        << "  QR Decomposition (Modified Gram-Schmidt)\n"
        << "=============================================\n\n"

        << "WHAT IS QR DECOMPOSITION?\n\n"
        << "  Any matrix A can be factored into:\n\n"
        << "    A = Q * R\n\n"
        << "  where:\n"
        << "    Q = orthonormal matrix (columns are unit-length and\n"
        << "        perpendicular to each other)\n"
        << "    R = upper-triangular matrix (zeros below the diagonal)\n\n"

        << "WHY DOES IT MATTER?\n\n"
        << "  1. LEAST-SQUARES REGRESSION\n"
        << "     Fitting a line (or curve) to noisy data.  Instead of\n"
        << "     solving A^T*A*x = A^T*b (numerically unstable), use:\n"
        << "       A = Q*R  ->  R*x = Q^T*b  (stable + fast)\n\n"
        << "  2. EIGENVALUE COMPUTATION\n"
        << "     The QR algorithm repeatedly applies QR decomposition\n"
        << "     to find all eigenvalues of a matrix.  This powers\n"
        << "     Google's PageRank, vibration analysis, and PCA.\n\n"
        << "  3. SOLVING LINEAR SYSTEMS\n"
        << "     For A*x = b:  decompose A = Q*R, then solve\n"
        << "     R*x = Q^T*b by back-substitution.  More stable\n"
        << "     than Gaussian elimination for ill-conditioned systems.\n\n"

        << "HOW MODIFIED GRAM-SCHMIDT WORKS\n\n"
        << "  Input: columns v1, v2, ..., vn of matrix A\n"
        << "  Goal:  produce orthonormal columns q1, q2, ..., qn\n\n"
        << "  For each column k:\n"
        << "    1. Compute its length  ->  R[k][k]\n"
        << "    2. Normalize it  (divide by length)  ->  qk\n"
        << "    3. For every later column j:\n"
        << "       a) Measure overlap with qk  ->  R[k][j]\n"
        << "       b) Subtract that overlap from column j\n\n"
        << "  Think of it like building perpendicular directions:\n"
        << "    - q1 = normalize(v1)\n"
        << "    - q2 = remove the q1 part from v2, then normalize\n"
        << "    - q3 = remove q1 and q2 parts from v3, then normalize\n\n"

        << "  Why \"Modified\" instead of \"Classical\"?\n"
        << "    Classical GS subtracts all projections at once.\n"
        << "    Modified GS updates remaining columns immediately\n"
        << "    after each normalization.  This keeps rounding errors\n"
        << "    from accumulating -- critical for real-world data.\n"
        << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Demo 1: Basic QR decomposition with verification
    // -----------------------------------------------------------------

    std::cout
        << "DEMO 1: Basic 3x3 Decomposition\n\n"
        << "  Decompose this matrix into Q * R:\n\n"
        << "    A = [ 1  1  0 ]\n"
        << "        [ 1  0  1 ]\n"
        << "        [ 0  1  1 ]\n\n";

    Matrix A = {
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };

    auto result = QrDecomposition<double>::Decompose(A);

    PrintMatrix("  Q (orthonormal):", result.Q);
    std::cout << "\n";
    PrintMatrix("  R (upper-triangular):", result.R);

    // Verify Q is orthonormal: Q^T * Q = I
    std::cout << "\n  Verify Q is orthonormal (Q^T * Q should be I):\n\n";
    std::size_t n = result.Q[0].size();
    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "    ";
        for (std::size_t j = 0; j < n; ++j) {
            double dot = ColumnDot(result.Q, i, j);
            std::cout << std::setw(8) << std::fixed
                      << std::setprecision(4) << dot << " ";
        }
        std::cout << "\n";
    }

    // Verify Q * R = A
    std::cout << "\n  Verify Q * R = A:\n";
    Matrix reconstructed = MultiplyMat(result.Q, result.R);
    PrintMatrix("", reconstructed);

    // -----------------------------------------------------------------
    // Demo 2: Least-Squares Regression
    //
    //   A company measures monthly ad spending vs revenue:
    //
    //     Month  Ads($k)  Revenue($k)
    //       1      1.0       2.1
    //       2      2.0       3.9
    //       3      3.0       6.2
    //       4      4.0       7.8
    //       5      5.0      10.1
    //
    //   Find the best-fit line:  revenue = a + b * ads
    //   This is an overdetermined system (5 equations, 2 unknowns).
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "DEMO 2: Least-Squares Regression\n"
              << "---------------------------------------------\n\n"
              << "  SCENARIO: A company tracks ad spending vs revenue\n\n"
              << "    Month   Ads ($k)   Revenue ($k)\n"
              << "      1       1.0          2.1\n"
              << "      2       2.0          3.9\n"
              << "      3       3.0          6.2\n"
              << "      4       4.0          7.8\n"
              << "      5       5.0         10.1\n\n"
              << "  Goal: Find best-fit line  revenue = a + b * ads\n\n"
              << "  This is an overdetermined system (5 equations, 2\n"
              << "  unknowns).  No exact solution exists because of\n"
              << "  measurement noise.  QR finds the \"closest\" answer.\n\n";

    // Build the system: A * x = b
    //   A = [1  t1]    x = [a]    b = [y1]
    //       [1  t2]        [b]        [y2]
    //       [...]                     [...]
    Vector ads  = {1.0, 2.0, 3.0, 4.0, 5.0};
    Vector revenue = {2.1, 3.9, 6.2, 7.8, 10.1};

    Matrix A_ls(ads.size(), Vector(2));
    for (std::size_t i = 0; i < ads.size(); ++i) {
        A_ls[i][0] = 1.0;       // constant term
        A_ls[i][1] = ads[i];    // coefficient of ads
    }

    std::cout << "  Step 1: Set up the matrix equation A * x = b\n\n"
              << "    A = [ 1  1.0 ]    x = [ a ]    b = [  2.1 ]\n"
              << "        [ 1  2.0 ]        [ b ]        [  3.9 ]\n"
              << "        [ 1  3.0 ]                     [  6.2 ]\n"
              << "        [ 1  4.0 ]                     [  7.8 ]\n"
              << "        [ 1  5.0 ]                     [ 10.1 ]\n\n";

    // Decompose A = Q * R
    auto ls_result = QrDecomposition<double>::Decompose(A_ls);

    std::cout << "  Step 2: Decompose A = Q * R\n\n";
    PrintMatrix("    Q (5x2 orthonormal):", ls_result.Q);
    std::cout << "\n";
    PrintMatrix("    R (2x2 upper-triangular):", ls_result.R);

    // Solve: R * x = Q^T * b
    std::cout << "\n  Step 3: Solve R * x = Q^T * b\n\n"
              << "    Since Q is orthonormal, Q^T * Q = I, so:\n"
              << "      A*x = b  ->  Q*R*x = b  ->  R*x = Q^T*b\n\n"
              << "    R is upper-triangular, so back-substitution gives\n"
              << "    the answer directly (no matrix inversion needed).\n\n";

    Matrix Qt = Transpose(ls_result.Q);
    Vector Qt_b = MatVecMul(Qt, revenue);
    Vector x = BackSubstitute(ls_result.R, Qt_b);

    PrintVector("    Q^T * b =", Qt_b);
    std::cout << "\n    Solving R * x = Q^T * b by back-substitution:\n\n";
    std::cout << "    a (intercept) = " << std::fixed << std::setprecision(4)
              << x[0] << "\n"
              << "    b (slope)     = " << x[1] << "\n\n"
              << "    Best-fit line: revenue = " << x[0] << " + "
              << x[1] << " * ads\n\n";

    // Show predictions
    std::cout << "  Step 4: Check predictions vs actual\n\n"
              << "    Ads($k)  Actual($k)  Predicted($k)  Error\n";
    for (std::size_t i = 0; i < ads.size(); ++i) {
        double predicted = x[0] + x[1] * ads[i];
        double error = revenue[i] - predicted;
        std::cout << "      " << std::fixed << std::setprecision(1)
                  << ads[i] << "       " << std::setprecision(1)
                  << revenue[i] << "         "
                  << std::setprecision(2) << predicted << "       "
                  << std::showpos << std::setprecision(2)
                  << error << std::noshowpos << "\n";
    }

    std::cout << "\n    Every $1k in ads -> ~$" << std::fixed
              << std::setprecision(2) << x[1]
              << "k more revenue.\n";

    std::cout << "\n  WHY QR INSTEAD OF NORMAL EQUATIONS?\n\n"
              << "    The textbook approach is: A^T*A*x = A^T*b\n"
              << "    But A^T*A squares the condition number of A.\n"
              << "    If A has condition number k, then A^T*A has\n"
              << "    condition number k^2 -- much more sensitive\n"
              << "    to rounding errors.\n\n"
              << "    QR avoids forming A^T*A entirely.  This matters\n"
              << "    for real-world data where columns can be nearly\n"
              << "    linearly dependent (multicollinearity).\n";

    // -----------------------------------------------------------------
    // Demo 3: Larger example -- a well-known test matrix
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "DEMO 3: Classic Test Matrix\n"
              << "---------------------------------------------\n\n"
              << "  The matrix from Cleve Moler's textbook:\n\n"
              << "    A = [ 12  -51    4 ]\n"
              << "        [  6  167  -68 ]\n"
              << "        [ -4   24  -41 ]\n\n";

    Matrix A3 = {
        { 12.0, -51.0,   4.0},
        {  6.0, 167.0, -68.0},
        { -4.0,  24.0, -41.0}
    };

    auto r3 = QrDecomposition<double>::Decompose(A3);
    PrintMatrix("  Q:", r3.Q);
    std::cout << "\n";
    PrintMatrix("  R:", r3.R);

    Matrix check3 = MultiplyMat(r3.Q, r3.R);
    std::cout << "\n  Q * R (should match A):\n";
    PrintMatrix("", check3);

    // -----------------------------------------------------------------
    // Edge case: linearly dependent columns
    // -----------------------------------------------------------------

    std::cout << "---------------------------------------------\n"
              << "EDGE CASE: Linearly Dependent Columns\n"
              << "---------------------------------------------\n\n"
              << "  A = [ 1  2 ]   Column 2 = 2 * Column 1\n"
              << "      [ 2  4 ]   -> linearly dependent -> no QR\n\n";
    {
        Matrix singular = {{1.0, 2.0}, {2.0, 4.0}};
        try {
            (void)QrDecomposition<double>::Decompose(singular);
            std::cout << "  (unexpected: no exception)\n";
        } catch (const std::runtime_error& e) {
            std::cout << "  Caught: \"" << e.what() << "\"\n";
        }
    }

    // -----------------------------------------------------------------
    // GPU / CUDA discussion
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "SCALING UP: GPU / CUDA ALTERNATIVES\n"
              << "---------------------------------------------\n\n"

              << "  This implementation uses Modified Gram-Schmidt,\n"
              << "  which is great for learning and small matrices.\n"
              << "  For large-scale problems, here's what changes:\n\n"

              << "  HOUSEHOLDER REFLECTIONS (CPU, preferred for production)\n"
              << "    Instead of projecting one column at a time,\n"
              << "    Householder reflects an entire column to zero out\n"
              << "    entries below the diagonal in one step.\n"
              << "    - Same O(m*n^2) but ~2x fewer floating-point ops\n"
              << "    - Better numerical stability than Gram-Schmidt\n"
              << "    - Used by LAPACK, Eigen, NumPy (np.linalg.qr)\n\n"

              << "  CUDA / GPU ACCELERATION\n"
              << "    Householder reflections are naturally parallelizable\n"
              << "    because each reflection is a matrix-vector multiply\n"
              << "    (BLAS Level 2/3), which GPUs excel at.\n\n"
              << "    cuSOLVER (NVIDIA's library) provides:\n"
              << "      cusolverDnDgeqrf()  -- QR factorization\n"
              << "      cusolverDnDormqr()  -- apply Q to a vector\n\n"
              << "    Typical speedups:\n"
              << "      100x100 matrix:    ~same as CPU (GPU overhead)\n"
              << "      1000x1000 matrix:  ~5-10x faster on GPU\n"
              << "      10000x10000:       ~50-100x faster on GPU\n\n"
              << "    The GPU advantage comes from thousands of cores\n"
              << "    doing the matrix multiplications in parallel.\n\n"

              << "  BLOCKED ALGORITHMS (multi-core CPU)\n"
              << "    Split the matrix into panels (blocks of columns).\n"
              << "    Process each panel with Householder, then update\n"
              << "    the trailing matrix using Level-3 BLAS (matrix-matrix\n"
              << "    multiply).  This maximizes cache utilization.\n"
              << "    Used by LAPACK's dgeqrf with block size ~32-64.\n\n"

              << "  COMPARISON TABLE\n\n"
              << "    Method             Best For         Stability\n"
              << "    ----------------   ---------------  ----------\n"
              << "    Gram-Schmidt       Learning, small  Good (MGS)\n"
              << "    Householder        Production CPU   Excellent\n"
              << "    Blocked Householder Multi-core CPU  Excellent\n"
              << "    cuSOLVER (GPU)     Large matrices   Excellent\n";

    // -----------------------------------------------------------------
    // Interactive mode
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Interactive Mode\n"
              << "  Enter a 2x2 matrix: a b c d  for [[a,b],[c,d]]\n"
              << "  Or 'q' to quit.\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  > ";
        std::string line;
        std::getline(std::cin, line);

        if (line.empty() || line[0] == 'q' || line[0] == 'Q') break;

        try {
            std::size_t p1 = 0;
            double a = std::stod(line, &p1);
            std::size_t p2 = 0;
            double b_val = std::stod(line.substr(p1), &p2);
            std::size_t p3 = 0;
            double c = std::stod(line.substr(p1 + p2), &p3);
            double d = std::stod(line.substr(p1 + p2 + p3));

            Matrix m = {{a, b_val}, {c, d}};
            std::cout << "\n    Input matrix:\n";
            PrintMatrix("", m);

            auto res = QrDecomposition<double>::Decompose(m);
            std::cout << "\n";
            PrintMatrix("    Q:", res.Q);
            std::cout << "\n";
            PrintMatrix("    R:", res.R);

            Matrix verify = MultiplyMat(res.Q, res.R);
            std::cout << "\n    Q * R (should match input):\n";
            PrintMatrix("", verify);
        } catch (const std::exception& e) {
            std::cout << "    Error: " << e.what()
                      << "\n    Enter 4 numbers: a b c d"
                      << "  (e.g. 3 1 4 1)\n";
        }
    }

    std::cout << "\nDone.\n";
    return EXIT_SUCCESS;
}
