/**
 * @file test_PostfixToInfix.cpp
 * @brief Unit tests for PostfixToInfix converter
 */

#include "PostfixToInfix.hpp"
#include <gtest/gtest.h>

TEST(PostfixToInfixTest, SimpleAddition) {
    EXPECT_EQ(PostfixToInfix::Convert("ab+"), "(a+b)");
}

TEST(PostfixToInfixTest, SimpleMultiplication) {
    EXPECT_EQ(PostfixToInfix::Convert("ab*"), "(a*b)");
}

TEST(PostfixToInfixTest, TwoOperators) {
    // ab*c+ = (a*b)+c
    EXPECT_EQ(PostfixToInfix::Convert("ab*c+"), "((a*b)+c)");
}

TEST(PostfixToInfixTest, ComplexExpression) {
    // abc/-ad/e-* = (a - b/c) * (a/d - e)
    std::string result = PostfixToInfix::Convert("abc/-ad/e-*");
    EXPECT_EQ(result, "((a-(b/c))*((a/d)-e))");
}

TEST(PostfixToInfixTest, SingleOperand) {
    // edge case: should return "a" as-is (but actually not valid postfix
    // for an expression). Our implementation pushes it and returns it.
    EXPECT_EQ(PostfixToInfix::Convert("a"), "a");
}

TEST(PostfixToInfixTest, EmptyInput) {
    EXPECT_EQ(PostfixToInfix::Convert(""), "");
}

TEST(PostfixToInfixTest, Exponentiation) {
    EXPECT_EQ(PostfixToInfix::Convert("ab^"), "(a^b)");
}

TEST(PostfixToInfixTest, NestedOperations) {
    // abcd+*- = a - (b * (c + d))
    EXPECT_EQ(PostfixToInfix::Convert("abcd+*-"), "(a-(b*(c+d)))");
}

TEST(PostfixToInfixTest, MalformedTooFewOperands) {
    EXPECT_THROW(PostfixToInfix::Convert("+"), std::invalid_argument);
    EXPECT_THROW(PostfixToInfix::Convert("a+*"), std::invalid_argument);
}

TEST(PostfixToInfixTest, MalformedTooManyOperands) {
    EXPECT_THROW(PostfixToInfix::Convert("abc"), std::invalid_argument);
}

TEST(PostfixToInfixTest, InvalidCharacter) {
    EXPECT_THROW(PostfixToInfix::Convert("a1+"), std::invalid_argument);
}

TEST(PostfixToInfixTest, UpperCaseOperands) {
    EXPECT_EQ(PostfixToInfix::Convert("AB+"), "(A+B)");
}
