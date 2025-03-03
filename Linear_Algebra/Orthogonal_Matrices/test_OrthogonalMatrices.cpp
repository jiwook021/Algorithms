/**
 * @file test_OrthogonalMatrices.cpp
 * @brief Unit tests for Gram-Schmidt and orthogonality utilities.
 * @details Verifies orthonormality of Gram-Schmidt output, orthogonal
 *          matrix detection, and vector operation correctness.
 */

#include "OrthogonalMatrices.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

static constexpr double TOL = 1e-9;

static bool Near(double a, double b, double eps = TOL) {
    return std::fabs(a - b) < eps;
}

void TestDotProduct() {
    assert(Near(OrthogonalMatrices::DotProduct({1, 0}, {0, 1}), 0.0));
    assert(Near(OrthogonalMatrices::DotProduct({3, 4}, {3, 4}), 25.0));
    std::cout << "  TestDotProduct             PASSED\n";
}

void TestNorm() {
    assert(Near(OrthogonalMatrices::Norm({3.0, 4.0}), 5.0));
    assert(Near(OrthogonalMatrices::Norm({0.0, 0.0}), 0.0));
    std::cout << "  TestNorm                   PASSED\n";
}

void TestScalarMultiply() {
    auto r = OrthogonalMatrices::ScalarMultiply({1.0, 2.0}, 3.0);
    assert(Near(r[0], 3.0) && Near(r[1], 6.0));
    std::cout << "  TestScalarMultiply         PASSED\n";
}

void TestSubtractVectors() {
    auto r = OrthogonalMatrices::SubtractVectors({5.0, 3.0}, {2.0, 1.0});
    assert(Near(r[0], 3.0) && Near(r[1], 2.0));
    std::cout << "  TestSubtractVectors        PASSED\n";
}

void TestGramSchmidtOrthonormality() {
    OrthogonalMatrices::Matrix input = {
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };
    auto basis = OrthogonalMatrices::GramSchmidt(input);
    assert(basis.size() == 3);

    // Check orthonormality: q_i . q_j = delta_{ij}.
    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t j = 0; j < basis.size(); ++j) {
            double dot = OrthogonalMatrices::DotProduct(basis[i], basis[j]);
            double expected = (i == j) ? 1.0 : 0.0;
            assert(Near(dot, expected));
        }
    }
    std::cout << "  TestGramSchmidtOrthonorm   PASSED\n";
}

void TestGramSchmidtLinearDependent() {
    // Third vector is a linear combination of first two.
    OrthogonalMatrices::Matrix input = {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}   // = v1 + v2
    };
    auto basis = OrthogonalMatrices::GramSchmidt(input);
    // Should only produce 2 basis vectors.
    assert(basis.size() == 2);
    std::cout << "  TestGramSchmidtLinDepend   PASSED\n";
}

void TestIsOrthogonalTrue() {
    OrthogonalMatrices::Matrix q = {
        {1.0,  0.0, 0.0},
        {0.0,  0.0, -1.0},
        {0.0,  1.0, 0.0}
    };
    assert(OrthogonalMatrices::IsOrthogonalMatrix(q));
    std::cout << "  TestIsOrthogonalTrue       PASSED\n";
}

void TestIsOrthogonalFalse() {
    OrthogonalMatrices::Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    assert(!OrthogonalMatrices::IsOrthogonalMatrix(m));
    std::cout << "  TestIsOrthogonalFalse      PASSED\n";
}

void TestIdentityIsOrthogonal() {
    OrthogonalMatrices::Matrix id = {{1, 0}, {0, 1}};
    assert(OrthogonalMatrices::IsOrthogonalMatrix(id));
    std::cout << "  TestIdentityIsOrthogonal   PASSED\n";
}

void TestGramSchmidtProducesOrthogonal() {
    // Build a matrix whose columns are the Gram-Schmidt output and
    // verify it is orthogonal.
    OrthogonalMatrices::Matrix input = {
        {1.0, 1.0, 0.0},
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    };
    auto basis = OrthogonalMatrices::GramSchmidt(input);

    // Basis vectors are rows; transpose to get columns.
    std::size_t n = basis.size();
    OrthogonalMatrices::Matrix q(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            q[i][j] = basis[j][i];

    assert(OrthogonalMatrices::IsOrthogonalMatrix(q));
    std::cout << "  TestGSProducesOrthogonal   PASSED\n";
}

int main() {
    std::cout << "Running OrthogonalMatrices tests...\n";
    TestDotProduct();
    TestNorm();
    TestScalarMultiply();
    TestSubtractVectors();
    TestGramSchmidtOrthonormality();
    TestGramSchmidtLinearDependent();
    TestIsOrthogonalTrue();
    TestIsOrthogonalFalse();
    TestIdentityIsOrthogonal();
    TestGramSchmidtProducesOrthogonal();
    std::cout << "All OrthogonalMatrices tests PASSED.\n";
    return EXIT_SUCCESS;
}
