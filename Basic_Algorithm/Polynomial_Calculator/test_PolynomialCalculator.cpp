/**
 * @file test_PolynomialCalculator.cpp
 * @brief Unit tests for PolynomialCalculator
 */

#include "PolynomialCalculator.hpp"
#include <gtest/gtest.h>

using namespace PolynomialCalculator;

TEST(PolynomialCalculator, RemoveSpacesBasic) {
    EXPECT_EQ(RemoveSpaces("3x + 2"), "3x+2");
}

TEST(PolynomialCalculator, RemoveSpacesMultiple) {
    EXPECT_EQ(RemoveSpaces("  a b c  "), "abc");
}

TEST(PolynomialCalculator, RemoveSpacesEmpty) {
    EXPECT_EQ(RemoveSpaces(""), "");
}

TEST(PolynomialCalculator, RemoveSpacesNoSpaces) {
    EXPECT_EQ(RemoveSpaces("abc"), "abc");
}

TEST(PolynomialCalculator, HasNumberTrue) {
    EXPECT_TRUE(HasNumber("3x"));
    EXPECT_TRUE(HasNumber("abc123"));
}

TEST(PolynomialCalculator, HasNumberFalse) {
    EXPECT_FALSE(HasNumber("xyz"));
    EXPECT_FALSE(HasNumber(""));
}

TEST(PolynomialCalculator, ContainsAsteriskTrue) {
    EXPECT_TRUE(ContainsAsterisk("3*x"));
}

TEST(PolynomialCalculator, ContainsAsteriskFalse) {
    EXPECT_FALSE(ContainsAsterisk("3x"));
    EXPECT_FALSE(ContainsAsterisk(""));
}

TEST(PolynomialCalculator, SplitPolynomialMultipleTerms) {
    auto parts = SplitPolynomial("3x+2x-1");
    ASSERT_EQ(parts.size(), 3u);
    EXPECT_EQ(parts[0], "3x");
    EXPECT_EQ(parts[1], "+2x");
    EXPECT_EQ(parts[2], "-1");
}

TEST(PolynomialCalculator, SplitPolynomialSingleTerm) {
    auto parts = SplitPolynomial("5x");
    ASSERT_EQ(parts.size(), 1u);
    EXPECT_EQ(parts[0], "5x");
}

TEST(PolynomialCalculator, SplitPolynomialLeadingNegative) {
    auto parts = SplitPolynomial("-3x+2");
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0], "-3x");
    EXPECT_EQ(parts[1], "+2");
}

TEST(PolynomialCalculator, SplitPolynomialEmpty) {
    auto parts = SplitPolynomial("");
    EXPECT_TRUE(parts.empty());
}
