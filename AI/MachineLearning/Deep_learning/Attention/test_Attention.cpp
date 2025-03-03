/**
 * @file test_Attention.cpp
 * @brief Unit tests for the Attention module.
 */
#include <gtest/gtest.h>
#include "Attention.hpp"
#include <cmath>
#include <numeric>

using namespace Ml;

TEST(MatMul, CorrectResult) {
    Matrix a = {{1, 2}, {3, 4}};
    Matrix b = {{5, 6}, {7, 8}};
    auto c = MatMul(a, b);
    EXPECT_DOUBLE_EQ(c[0][0], 19.0);
    EXPECT_DOUBLE_EQ(c[1][1], 50.0);
}

TEST(MatMul, DimMismatchThrows) {
    Matrix a = {{1, 2}};
    Matrix b = {{1, 2}};
    EXPECT_THROW(MatMul(a, b), std::invalid_argument);
}

TEST(Transpose, CorrectDimensions) {
    Matrix m = {{1, 2, 3}, {4, 5, 6}};
    auto t = Transpose(m);
    EXPECT_EQ(t.size(), 3u);
    EXPECT_EQ(t[0].size(), 2u);
    EXPECT_DOUBLE_EQ(t[2][1], 6.0);
}

TEST(Softmax, SumsToOne) {
    auto s = Softmax({1.0, 2.0, 3.0});
    double sum = std::accumulate(s.begin(), s.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-9);
}

TEST(Softmax, MaxElementGetsHighestProb) {
    auto s = Softmax({1.0, 5.0, 2.0});
    EXPECT_GT(s[1], s[0]);
    EXPECT_GT(s[1], s[2]);
}

TEST(Attention, OutputSize) {
    Matrix q = {{1,0,1,0}, {0,1,0,1}};
    Matrix k = {{1,0,1,0}, {0,1,0,1}, {1,1,0,0}};
    Matrix v = {{1,2,3}, {4,5,6}, {7,8,9}};
    ScaledDotProductAttention attn;
    auto out = attn.ComputeAttention(q, k, v);
    EXPECT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].size(), 3u);
}

TEST(Attention, RowsSumToOne) {
    Matrix q = {{1,0}, {0,1}};
    Matrix k = {{1,0}, {0,1}};
    Matrix v = {{1,2}, {3,4}};
    ScaledDotProductAttention attn;
    auto out = attn.ComputeAttention(q, k, v);
    for (const auto& row : out) {
        double sum = std::accumulate(row.begin(), row.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-9);
    }
}

TEST(Attention, EmptyInputThrows) {
    ScaledDotProductAttention attn;
    EXPECT_THROW(attn.ComputeAttention({}, {{1}}, {{1}}), std::invalid_argument);
}
