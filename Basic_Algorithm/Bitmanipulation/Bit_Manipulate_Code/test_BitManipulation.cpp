/**
 * @file test_BitManipulation.cpp
 * @brief Unit tests for BitManipulation
 */

#include "BitManipulation.hpp"
#include <gtest/gtest.h>

using namespace BitManipulation;

TEST(BitManipulation, ReverseBitBasic) {
    EXPECT_EQ(static_cast<unsigned char>(ReverseBit(static_cast<char>(0b10110000))),
              static_cast<unsigned char>(0b00001101));
}

TEST(BitManipulation, ReverseBitZero) {
    EXPECT_EQ(ReverseBit(0), 0);
}

TEST(BitManipulation, BitExtractBasic) {
    EXPECT_EQ(BitExtracted(0b11010110, 3, 2), 0b011);
}

TEST(BitManipulation, SwapBitsBasic) {
    // 28 = 0b11100: bit0=0, bit3=1 -> swap -> 0b10101 = 21
    EXPECT_EQ(SwapBits(28, 0, 3), 21u);
}

TEST(BitManipulation, SwapBitsSameValue) {
    // Swapping identical bits should not change value
    EXPECT_EQ(SwapBits(0b1111, 0, 3), 0b1111u);
}

TEST(BitManipulation, ReverseDigitsBasic) {
    EXPECT_EQ(ReverseDigits(1234), 4321);
}

TEST(BitManipulation, ReverseDigitsZero) {
    EXPECT_EQ(ReverseDigits(0), 0);
}

TEST(BitManipulation, ReverseDigitsSingle) {
    EXPECT_EQ(ReverseDigits(5), 5);
}

TEST(BitManipulation, IsPalindromeTrue) {
    EXPECT_TRUE(IsPalindrome(121));
    EXPECT_TRUE(IsPalindrome(12321));
    EXPECT_TRUE(IsPalindrome(0));
}

TEST(BitManipulation, IsPalindromeFalse) {
    EXPECT_FALSE(IsPalindrome(123));
    EXPECT_FALSE(IsPalindrome(-121));
}

TEST(BitManipulation, PowerBasic) {
    EXPECT_DOUBLE_EQ(Power(2, 10), 1024.0);
}

TEST(BitManipulation, PowerZeroExponent) {
    EXPECT_DOUBLE_EQ(Power(5, 0), 1.0);
}

TEST(BitManipulation, PowerNegativeExponent) {
    EXPECT_DOUBLE_EQ(Power(2, -1), 0.5);
}

TEST(BitManipulation, IntersectOverlapping) {
    Rectangle r1 = {0, 0, 4, 4};
    Rectangle r2 = {2, 2, 4, 4};
    Rectangle result;
    EXPECT_TRUE(Intersect(r1, r2, result));
    EXPECT_EQ(result.X, 2);
    EXPECT_EQ(result.Y, 2);
    EXPECT_EQ(result.Width, 2);
    EXPECT_EQ(result.Height, 2);
}

TEST(BitManipulation, IntersectNoOverlap) {
    Rectangle r1 = {0, 0, 2, 2};
    Rectangle r2 = {5, 5, 2, 2};
    Rectangle result;
    EXPECT_FALSE(Intersect(r1, r2, result));
}
