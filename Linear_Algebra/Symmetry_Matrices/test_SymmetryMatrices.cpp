/**
 * @file test_SymmetryMatrices.cpp
 * @brief Unit tests for SymmetryMatrix class.
 * @details Verifies symmetry checks, trace, determinant, Frobenius norm,
 *          positive-definiteness, addition, multiplication commutativity,
 *          and eigenvalue approximation.
 */

#include "SymmetryMatrices.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

static constexpr double TOL = 1e-9;

static bool Near(double a, double b, double eps = TOL) {
    return std::fabs(a - b) < eps;
}

void TestIsSymmetricTrue() {
    SymmetryMatrix m({{1, 2}, {2, 3}});
    assert(m.IsSymmetric());
    std::cout << "  TestIsSymmetricTrue        PASSED\n";
}

void TestIsSymmetricFalse() {
    SymmetryMatrix m({{1, 2}, {3, 4}});
    assert(!m.IsSymmetric());
    std::cout << "  TestIsSymmetricFalse       PASSED\n";
}

void TestTrace() {
    SymmetryMatrix m({{1, 0}, {0, 4}});
    assert(Near(m.Trace(), 5.0));
    SymmetryMatrix m3({{3, 1, 1}, {1, 5, 1}, {1, 1, 7}});
    assert(Near(m3.Trace(), 15.0));
    std::cout << "  TestTrace                  PASSED\n";
}

void TestDeterminant1x1() {
    SymmetryMatrix m({{42.0}});
    assert(Near(m.Determinant(), 42.0));
    std::cout << "  TestDeterminant1x1         PASSED\n";
}

void TestDeterminant2x2() {
    SymmetryMatrix m({{4, 2}, {2, 3}});
    // det = 12 - 4 = 8
    assert(Near(m.Determinant(), 8.0));
    std::cout << "  TestDeterminant2x2         PASSED\n";
}

void TestDeterminant3x3() {
    SymmetryMatrix m({{2, 1, 0}, {1, 3, 1}, {0, 1, 2}});
    // det = 2*(6-1) - 1*(2-0) + 0 = 10 - 2 = 8
    assert(Near(m.Determinant(), 8.0));
    std::cout << "  TestDeterminant3x3         PASSED\n";
}

void TestFrobeniusNorm() {
    SymmetryMatrix m({{1, 0}, {0, 1}});
    assert(Near(m.FrobeniusNorm(), std::sqrt(2.0)));
    std::cout << "  TestFrobeniusNorm          PASSED\n";
}

void TestPositiveDefinite() {
    // Positive definite: [[2,1],[1,2]]
    SymmetryMatrix pd({{2, 1}, {1, 2}});
    assert(pd.IsPositiveDefinite());
    // Not positive definite: [[-1,0],[0,1]]
    SymmetryMatrix npd({{-1, 0}, {0, 1}});
    assert(!npd.IsPositiveDefinite());
    std::cout << "  TestPositiveDefinite       PASSED\n";
}

void TestAdditionSymmetric() {
    SymmetryMatrix a({{1, 2}, {2, 3}});
    SymmetryMatrix b({{4, 1}, {1, 5}});
    auto sum = a.Add(b);
    assert(sum.IsSymmetric());
    assert(Near(sum.GetValue(0, 0), 5.0));
    assert(Near(sum.GetValue(0, 1), 3.0));
    assert(Near(sum.GetValue(1, 1), 8.0));
    std::cout << "  TestAdditionSymmetric      PASSED\n";
}

void TestMultiplyWithIdentity() {
    SymmetryMatrix a({{3, 1}, {1, 5}});
    SymmetryMatrix id = CreateIdentityMatrix(2);
    auto product = a.Multiply(id);
    assert(Near(product[0][0], 3.0));
    assert(Near(product[0][1], 1.0));
    assert(Near(product[1][0], 1.0));
    assert(Near(product[1][1], 5.0));
    std::cout << "  TestMultiplyWithIdentity   PASSED\n";
}

void TestCommutativeWithIdentity() {
    SymmetryMatrix a({{3, 1}, {1, 5}});
    SymmetryMatrix id = CreateIdentityMatrix(2);
    assert(a.IsCommutative(id));
    assert(a.IsProductSymmetric(id));
    std::cout << "  TestCommutativeWithId      PASSED\n";
}

void TestApproxEigenvalue() {
    // Identity matrix: largest eigenvalue should be 1.
    SymmetryMatrix id = CreateIdentityMatrix(3);
    double eigenval = id.ApproximateLargestEigenvalue();
    assert(Near(eigenval, 1.0, 1e-4));

    // [[5,0],[0,2]]: largest eigenvalue = 5
    SymmetryMatrix diag({{5, 0}, {0, 2}});
    double ev = diag.ApproximateLargestEigenvalue();
    assert(Near(ev, 5.0, 1e-4));
    std::cout << "  TestApproxEigenvalue       PASSED\n";
}

void TestSingleElement() {
    SymmetryMatrix m({{42.0}});
    assert(m.IsSymmetric());
    assert(Near(m.Trace(), 42.0));
    assert(Near(m.Determinant(), 42.0));
    std::cout << "  TestSingleElement          PASSED\n";
}

void TestRandomConstructor() {
    SymmetryMatrix m(3, 12345);
    assert(m.IsSymmetric());
    assert(m.GetSize() == 3);
    std::cout << "  TestRandomConstructor      PASSED\n";
}

int main() {
    std::cout << "Running SymmetryMatrices tests...\n";
    TestIsSymmetricTrue();
    TestIsSymmetricFalse();
    TestTrace();
    TestDeterminant1x1();
    TestDeterminant2x2();
    TestDeterminant3x3();
    TestFrobeniusNorm();
    TestPositiveDefinite();
    TestAdditionSymmetric();
    TestMultiplyWithIdentity();
    TestCommutativeWithIdentity();
    TestApproxEigenvalue();
    TestSingleElement();
    TestRandomConstructor();
    std::cout << "All SymmetryMatrices tests PASSED.\n";
    return EXIT_SUCCESS;
}
