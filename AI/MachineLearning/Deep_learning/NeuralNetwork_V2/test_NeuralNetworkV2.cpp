/**
 * @file test_NeuralNetworkV2.cpp
 * @brief Unit tests for the NeuralNetworkV2 module (Matrix, Vector, LR, MLP).
 */
#include <gtest/gtest.h>
#include "NeuralNetworkV2.hpp"
#include <cmath>

using namespace Ml;

// ======================== Matrix ========================

TEST(Matrix, ConstructionAndAccess) {
    Matrix m(2, 3);
    m.At(0, 0) = 5.0;
    EXPECT_DOUBLE_EQ(m.At(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(m.At(1, 2), 0.0);
}

TEST(Matrix, OutOfRangeThrows) {
    Matrix m(2, 2);
    EXPECT_THROW(m.At(2, 0), std::out_of_range);
}

TEST(Matrix, Transpose) {
    Matrix m({{1, 2, 3}, {4, 5, 6}});
    Matrix t = m.Transpose();
    EXPECT_EQ(t.GetRows(), 3u);
    EXPECT_EQ(t.GetCols(), 2u);
    EXPECT_DOUBLE_EQ(t.At(2, 1), 6.0);
}

TEST(Matrix, Addition) {
    Matrix a({{1, 2}, {3, 4}});
    Matrix b({{5, 6}, {7, 8}});
    Matrix c = a + b;
    EXPECT_DOUBLE_EQ(c.At(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(c.At(1, 1), 12.0);
}

TEST(Matrix, MultiplicationDimMismatchThrows) {
    Matrix a(2, 3);
    Matrix b(2, 3);
    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(Matrix, Multiplication) {
    Matrix a({{1, 2}, {3, 4}});
    Matrix b({{5, 6}, {7, 8}});
    Matrix c = a * b;
    EXPECT_DOUBLE_EQ(c.At(0, 0), 19.0);
    EXPECT_DOUBLE_EQ(c.At(1, 1), 50.0);
}

TEST(Matrix, Identity) {
    Matrix id = Matrix::Identity(3);
    EXPECT_DOUBLE_EQ(id.At(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(id.At(0, 1), 0.0);
}

TEST(Matrix, SigmoidBounds) {
    Matrix m({{0.0, 100.0, -100.0}});
    Matrix s = m.Sigmoid();
    EXPECT_NEAR(s.At(0, 0), 0.5, 1e-6);
    EXPECT_GT(s.At(0, 1), 0.99);
    EXPECT_LT(s.At(0, 2), 0.01);
}

// ======================== Vector ========================

TEST(Vector, DotProduct) {
    Vector a({1, 2, 3});
    Vector b({4, 5, 6});
    EXPECT_DOUBLE_EQ(a.Dot(b), 32.0);
}

TEST(Vector, Norm) {
    Vector v({3, 4});
    EXPECT_DOUBLE_EQ(v.Norm(), 5.0);
}

TEST(Vector, NormalizeZeroThrows) {
    Vector v(3);
    EXPECT_THROW(v.Normalize(), std::runtime_error);
}

TEST(Vector, ToMatrixRoundTrip) {
    Vector v({1, 2, 3});
    Matrix m = v.ToMatrix();
    Vector v2 = Vector::FromMatrix(m);
    EXPECT_EQ(v2.GetSize(), 3u);
    EXPECT_DOUBLE_EQ(v2.At(1), 2.0);
}

// ======================== LinearRegression ========================

TEST(LinearRegression, FitAndPredict) {
    // y = 2*x + 1
    std::vector<Vector> x;
    std::vector<double> y;
    for (int i = 0; i < 100; ++i) {
        x.push_back(Vector(std::vector<double>{static_cast<double>(i) / 10.0}));
        y.push_back(2.0 * (i / 10.0) + 1.0);
    }
    LinearRegression lr(1, 0.01, 2000);
    lr.Fit(x, y);
    double pred = lr.Predict(Vector(std::vector<double>{5.0}));
    EXPECT_NEAR(pred, 11.0, 1.0);
}

// ======================== LogisticRegression ========================

TEST(LogisticRegression, PredictProbaInRange) {
    LogisticRegression lr(2, 0.1, 100);
    double p = lr.PredictProba(Vector({1.0, 2.0}));
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
}

TEST(LogisticRegression, FitSeparableData) {
    std::vector<Vector> x;
    std::vector<int> y;
    for (int i = 0; i < 50; ++i) {
        x.push_back(Vector(std::vector<double>{static_cast<double>(i) / 10.0}));
        y.push_back(i < 25 ? 0 : 1);
    }
    LogisticRegression lr(1, 0.1, 1000);
    lr.Fit(x, y);
    EXPECT_EQ(lr.Predict(Vector(std::vector<double>{0.0})), 0);
    EXPECT_EQ(lr.Predict(Vector(std::vector<double>{4.9})), 1);
}

// ======================== NeuralNetworkV2 ========================

TEST(NeuralNetworkV2, ConstructionMinLayers) {
    EXPECT_THROW(NeuralNetworkV2({2}), std::invalid_argument);
    EXPECT_NO_THROW(NeuralNetworkV2({2, 1}));
}

TEST(NeuralNetworkV2, ForwardOutputSize) {
    NeuralNetworkV2 nn({2, 4, 1});
    Matrix x({{0.5, 0.5}});
    Matrix out = nn.Forward(x);
    EXPECT_EQ(out.GetRows(), 1u);
    EXPECT_EQ(out.GetCols(), 1u);
}

TEST(NeuralNetworkV2, ForwardOutputInRange) {
    NeuralNetworkV2 nn({2, 4, 1});
    Matrix x({{0.5, 0.5}});
    Matrix out = nn.Forward(x);
    EXPECT_GE(out.At(0, 0), 0.0);
    EXPECT_LE(out.At(0, 0), 1.0);
}
