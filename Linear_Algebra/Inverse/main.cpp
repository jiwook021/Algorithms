/**
 * @file main.cpp
 * @brief Interactive driver for matrix inversion.
 *
 * Shows what a matrix inverse is, why it matters (solving
 * systems of equations), walks through a concrete example,
 * and demonstrates edge cases.
 */

#include "Inverse.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Multiply matrix A by vector b, return the result vector.
 */
static std::vector<double> MultiplyMatVec(
        const Matrix<double>& A, const std::vector<double>& b) {
    std::vector<double> result(A.Rows(), 0.0);
    for (std::size_t i = 0; i < A.Rows(); ++i) {
        for (std::size_t j = 0; j < A.Cols(); ++j) {
            result[i] += A.At(i, j) * b[j];
        }
    }
    return result;
}

/**
 * @brief Multiply two matrices, return the result.
 */
static Matrix<double> MultiplyMat(
        const Matrix<double>& A, const Matrix<double>& B) {
    Matrix<double> C(A.Rows(), B.Cols());
    for (std::size_t i = 0; i < A.Rows(); ++i) {
        for (std::size_t j = 0; j < B.Cols(); ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < A.Cols(); ++k) {
                sum += A.At(i, k) * B.At(k, j);
            }
            C.At(i, j) = sum;
        }
    }
    return C;
}

static void PrintVec(const std::vector<double>& v) {
    std::cout << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(2) << v[i];
    }
    std::cout << "]";
}

int main() {
    std::cout << "=============================================\n"
              << "  Matrix Inverse (Gauss-Jordan Elimination)\n"
              << "=============================================\n\n"

              << "PURPOSE\n"
              << "  The inverse of a matrix A is a matrix A^{-1} such\n"
              << "  that A * A^{-1} = I (the identity matrix).  It lets\n"
              << "  you solve systems of linear equations:\n\n"
              << "    Ax = b  -->  x = A^{-1} * b\n\n"

              << "HOW IT WORKS (Gauss-Jordan)\n"
              << "  1. Build the augmented matrix [A | I].\n"
              << "  2. For each column:\n"
              << "     a) Find the best pivot (largest absolute value).\n"
              << "     b) Swap it into the diagonal position.\n"
              << "     c) Scale the row so the pivot becomes 1.\n"
              << "     d) Subtract multiples of this row from all\n"
              << "        others to zero out the rest of the column.\n"
              << "  3. When the left half becomes I, the right half\n"
              << "     is A^{-1}.\n\n"

              << "WHY PARTIAL PIVOTING?\n"
              << "  Always picking the largest available value as the\n"
              << "  pivot reduces rounding errors.  Dividing by a tiny\n"
              << "  number amplifies errors; dividing by the largest\n"
              << "  available number keeps them small.\n"
              << "---------------------------------------------\n\n";

    // -----------------------------------------------------------------
    // Scenario: Solving a system of equations
    //
    //   A store sells apples and oranges.
    //   Two customers bought:
    //     Customer A: 3 apples + 2 oranges = $16
    //     Customer B: 1 apple  + 4 oranges = $12
    //
    //   What does each fruit cost?
    //   This is the system: A * x = b
    //     A = [[3, 2], [1, 4]]   b = [16, 12]
    //     x = A^{-1} * b = [price_apple, price_orange]
    // -----------------------------------------------------------------

    std::cout << "SCENARIO: Solving a System of Equations\n\n"
              << "  A store sells apples and oranges.  Two customers:\n\n"
              << "    Customer A:  3 apples + 2 oranges = $16\n"
              << "    Customer B:  1 apple  + 4 oranges = $12\n\n"
              << "  What does each fruit cost?\n\n"
              << "  In matrix form:  A * x = b\n\n"
              << "    A = [ 3  2 ]    b = [ 16 ]    x = [ apple  ]\n"
              << "        [ 1  4 ]        [ 12 ]        [ orange ]\n\n";

    Matrix<double> A({{3.0, 2.0}, {1.0, 4.0}});
    std::vector<double> b = {16.0, 12.0};

    std::cout << "  Step 1: Compute A^{-1}\n\n";
    Matrix<double> A_inv = A.Inverse();
    std::cout << "    A^{-1} =\n";
    A_inv.Print();

    std::cout << "\n  Step 2: Verify A * A^{-1} = I\n\n";
    Matrix<double> product = MultiplyMat(A, A_inv);
    std::cout << "    A * A^{-1} =\n";
    product.Print();

    std::cout << "\n  Step 3: Solve x = A^{-1} * b\n\n";
    std::vector<double> x = MultiplyMatVec(A_inv, b);
    std::cout << "    x = A^{-1} * b = ";
    PrintVec(x);
    std::cout << "\n\n"
              << "    Apple  = $" << std::fixed << std::setprecision(2)
              << x[0] << "\n"
              << "    Orange = $" << x[1] << "\n\n";

    std::cout << "  Step 4: Verify solution satisfies both equations\n\n";
    std::vector<double> check = MultiplyMatVec(A, x);
    std::cout << "    3*" << x[0] << " + 2*" << x[1] << " = "
              << check[0] << "  (should be 16)\n"
              << "    1*" << x[0] << " + 4*" << x[1] << " = "
              << check[1] << "  (should be 12)\n";

    // -----------------------------------------------------------------
    // 3x3 Example
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  3x3 Example\n"
              << "---------------------------------------------\n\n"
              << "  A larger system with 3 unknowns:\n\n"
              << "    2x + 1y + 1z =  8\n"
              << "    1x + 3y + 2z = 14\n"
              << "    1x + 0y + 2z =  6\n\n";

    Matrix<double> A3({{2.0, 1.0, 1.0},
                       {1.0, 3.0, 2.0},
                       {1.0, 0.0, 2.0}});
    std::vector<double> b3 = {8.0, 14.0, 6.0};

    Matrix<double> A3_inv = A3.Inverse();
    std::cout << "  A^{-1} =\n";
    A3_inv.Print();

    std::vector<double> x3 = MultiplyMatVec(A3_inv, b3);
    std::cout << "\n  Solution: x=";
    PrintVec(x3);
    std::cout << "\n\n  Verify: A * x = ";
    std::vector<double> check3 = MultiplyMatVec(A3, x3);
    PrintVec(check3);
    std::cout << "  (should be [8, 14, 6])\n";

    // -----------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Edge Cases\n"
              << "---------------------------------------------\n\n";

    // Identity is its own inverse
    std::cout << "  1. Identity matrix (its own inverse):\n\n";
    {
        Matrix<double> I({{1.0, 0.0}, {0.0, 1.0}});
        Matrix<double> I_inv = I.Inverse();
        std::cout << "     I^{-1} =\n";
        I_inv.Print();
    }

    // 1x1 matrix
    std::cout << "\n  2. 1x1 matrix:  [[5]]^{-1} = ";
    {
        Matrix<double> tiny({{5.0}});
        Matrix<double> tiny_inv = tiny.Inverse();
        std::cout << "[[" << tiny_inv.At(0, 0) << "]]  (1/5 = 0.2)\n";
    }

    // Singular matrix
    std::cout << "\n  3. Singular matrix (no inverse exists):\n\n"
              << "     [ 1  2 ]   Row 2 is 2x Row 1 -> linearly dependent\n"
              << "     [ 2  4 ]   determinant = 0 -> no inverse\n\n";
    {
        Matrix<double> singular({{1.0, 2.0}, {2.0, 4.0}});
        try {
            (void)singular.Inverse();
            std::cout << "     (unexpected: no exception)\n";
        } catch (const std::runtime_error& e) {
            std::cout << "     Caught: \"" << e.what() << "\"\n";
        }
    }

    // Non-square matrix
    std::cout << "\n  4. Non-square matrix (2x3 -- cannot invert):\n\n";
    {
        Matrix<double> rect(2, 3);
        try {
            (void)rect.Inverse();
            std::cout << "     (unexpected: no exception)\n";
        } catch (const std::runtime_error& e) {
            std::cout << "     Caught: \"" << e.what() << "\"\n";
        }
    }

    // -----------------------------------------------------------------
    // Interactive mode
    // -----------------------------------------------------------------

    std::cout << "\n---------------------------------------------\n"
              << "  Interactive Mode (enter a 2x2 matrix)\n"
              << "  Enter 4 numbers: a b c d  for [[a,b],[c,d]]\n"
              << "  Or 'q' to quit.\n"
              << "---------------------------------------------\n";

    while (true) {
        std::cout << "\n  > ";
        std::string line;
        std::getline(std::cin, line);

        if (line.empty() || line[0] == 'q' || line[0] == 'Q') break;

        try {
            double a, b_val, c, d;
            std::size_t pos = 0;
            a = std::stod(line, &pos);
            b_val = std::stod(line.substr(pos), &pos);
            // pos is relative to the substring, need cumulative offset
            std::string rest = line.substr(line.size() - (line.substr(
                line.find_first_not_of(" \t", line.find_first_of(
                    "0123456789.eE+-", pos))).size()));
            // Simpler: just parse all 4 from the line
            std::size_t p1 = 0;
            a = std::stod(line, &p1);
            std::size_t p2 = 0;
            b_val = std::stod(line.substr(p1), &p2);
            std::size_t p3 = 0;
            c = std::stod(line.substr(p1 + p2), &p3);
            d = std::stod(line.substr(p1 + p2 + p3));

            Matrix<double> m({{a, b_val}, {c, d}});
            std::cout << "    Matrix:\n";
            m.Print();

            Matrix<double> inv = m.Inverse();
            std::cout << "    Inverse:\n";
            inv.Print();

            Matrix<double> verify = MultiplyMat(m, inv);
            std::cout << "    A * A^{-1} (should be I):\n";
            verify.Print();
        } catch (const std::exception& e) {
            std::cout << "    Error: " << e.what()
                      << "\n    Enter 4 numbers: a b c d"
                      << "  (e.g. 4 7 2 6)\n";
        }
    }

    std::cout << "\nDone.\n";
    return EXIT_SUCCESS;
}
