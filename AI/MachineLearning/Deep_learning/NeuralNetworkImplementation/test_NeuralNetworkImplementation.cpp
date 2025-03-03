/**
 * @file test_NeuralNetworkImplementation.cpp
 * @brief Unit tests for NeuralNetworkImplementation (Vector, Matrix, GD).
 */
#include <gtest/gtest.h>
#include "NeuralNetworkImplementation.hpp"
#include <cmath>

using namespace Ml;

TEST(Vector, SizeAndAccess) {
    Vector v(3);
    v[0] = 1.0; v[1] = 2.0; v[2] = 3.0;
    EXPECT_EQ(v.Size(), 3u);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
}

TEST(Vector, DotProduct) {
    Vector a({1, 2, 3});
    Vector b({4, 5, 6});
    EXPECT_DOUBLE_EQ(a.Dot(b), 32.0);
}

TEST(Vector, DotMismatchThrows) {
    Vector a(2); Vector b(3);
    EXPECT_THROW(a.Dot(b), std::invalid_argument);
}

TEST(Vector, AddAndScale) {
    Vector a({1, 2});
    Vector b({3, 4});
    Vector c = a + b;
    EXPECT_DOUBLE_EQ(c[0], 4.0);
    Vector d = a * 2.0;
    EXPECT_DOUBLE_EQ(d[1], 4.0);
}

TEST(Matrix, DimensionsAndAccess) {
    Matrix m(2, 3);
    m(0, 2) = 7.0;
    EXPECT_EQ(m.GetRows(), 2u);
    EXPECT_EQ(m.GetCols(), 3u);
    EXPECT_DOUBLE_EQ(m(0, 2), 7.0);
}

TEST(Matrix, Transpose) {
    Matrix m(2, 3);
    m(0, 1) = 5.0;
    Matrix t = m.Transpose();
    EXPECT_EQ(t.GetRows(), 3u);
    EXPECT_DOUBLE_EQ(t(1, 0), 5.0);
}

TEST(Matrix, Multiply) {
    Matrix a(2, 2); a(0,0)=1; a(0,1)=2; a(1,0)=3; a(1,1)=4;
    Matrix b(2, 2); b(0,0)=5; b(0,1)=6; b(1,0)=7; b(1,1)=8;
    Matrix c = a * b;
    EXPECT_DOUBLE_EQ(c(0, 0), 19.0);
    EXPECT_DOUBLE_EQ(c(1, 1), 50.0);
}

TEST(Matrix, MultiplyDimMismatchThrows) {
    Matrix a(2, 3); Matrix b(2, 3);
    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(Matrix, MatVecMultiply) {
    Matrix m(2, 2); m(0,0)=1; m(0,1)=2; m(1,0)=3; m(1,1)=4;
    Vector v({5, 6});
    Vector r = m * v;
    EXPECT_DOUBLE_EQ(r[0], 17.0);
    EXPECT_DOUBLE_EQ(r[1], 39.0);
}

TEST(LinearRegressionGD, RecoversTrueWeights) {
    // y = 0.5*x1 + 1.5*x2
    Matrix x(3, 2);
    x(0,0)=1; x(0,1)=2; x(1,0)=3; x(1,1)=4; x(2,0)=5; x(2,1)=6;
    Vector y({3.5, 7.5, 11.5});

    Vector w = LinearRegressionGD(x, y, 0.01, 1000);
    EXPECT_NEAR(w[0], 0.5, 0.1);
    EXPECT_NEAR(w[1], 1.5, 0.1);
}
