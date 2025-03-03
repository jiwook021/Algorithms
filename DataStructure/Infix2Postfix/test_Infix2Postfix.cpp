/**
 * @file infix2postfix_test.cpp
 * @brief Unit tests for Shunting-Yard converter (infix2postfix.hpp).
 *
 * Compile:
 *   g++ -std=c++23 -Wall -Wextra infix2postfix_test.cpp -lgtest -lgtest_main -pthread -o infix2postfix_test
 */

#include "Infix2Postfix.hpp"

#include <gtest/gtest.h>

// =============================================================================
// Google Test
// =============================================================================

using ShuntingYard::ToPostfix;
using ShuntingYard::EvaluatePostfix;

TEST(Infix2PostfixTest, SimpleAddition) {
    EXPECT_EQ(ToPostfix("A+B"), "AB+");
}

TEST(Infix2PostfixTest, PrecedenceMulOverAdd) {
    EXPECT_EQ(ToPostfix("A+B*C"), "ABC*+");
}

TEST(Infix2PostfixTest, Parentheses) {
    EXPECT_EQ(ToPostfix("(A+B)*C"), "AB+C*");
}

TEST(Infix2PostfixTest, NestedParentheses) {
    EXPECT_EQ(ToPostfix("A*(B+C*D)+E"), "ABCD*+*E+");
}

TEST(Infix2PostfixTest, AllOperators) {
    EXPECT_EQ(ToPostfix("A+B-C*D/E"), "AB+CD*E/-");
}

TEST(Infix2PostfixTest, RightAssociativeExponent) {
    // 2^3^2 should be 2^(3^2) = 2^9, not (2^3)^2 = 8^2
    EXPECT_EQ(ToPostfix("A^B^C"), "ABC^^");
}

TEST(Infix2PostfixTest, ComplexExpression) {
    EXPECT_EQ(ToPostfix("((A+B)*C-D)*E"), "AB+C*D-E*");
}

TEST(Infix2PostfixTest, SingleOperand) {
    EXPECT_EQ(ToPostfix("A"), "A");
}

TEST(Infix2PostfixTest, WhitespaceIgnored) {
    EXPECT_EQ(ToPostfix("A + B * C"), "ABC*+");
}

TEST(Infix2PostfixTest, MismatchedRightParen) {
    EXPECT_THROW(ToPostfix("A+B)"), std::invalid_argument);
}

TEST(Infix2PostfixTest, MismatchedLeftParen) {
    EXPECT_THROW(ToPostfix("(A+B"), std::invalid_argument);
}

TEST(Infix2PostfixTest, EvaluateSimple) {
    std::string Postfix = ToPostfix("3+4*2");
    EXPECT_EQ(EvaluatePostfix(Postfix), 11);  // 3 + (4*2) = 11
}

TEST(Infix2PostfixTest, EvaluateParentheses) {
    std::string Postfix = ToPostfix("(3+4)*2");
    EXPECT_EQ(EvaluatePostfix(Postfix), 14);  // (3+4)*2 = 14
}

TEST(Infix2PostfixTest, EvaluateExponent) {
    std::string Postfix = ToPostfix("2^3");
    EXPECT_EQ(EvaluatePostfix(Postfix), 8);
}
