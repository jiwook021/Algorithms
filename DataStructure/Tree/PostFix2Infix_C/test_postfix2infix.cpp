#include <gtest/gtest.h>
#include <string>
#include <cstring>
#include <cstdlib>

// Adapted from PostFix2Infix_C - convert postfix to infix
static int isOperand(char x) { return (x >= 'a' && x <= 'z') || (x >= 'A' && x <= 'Z'); }

std::string Postfix2Infix(const std::string& postfix) {
    std::string stack[100]; int top = -1;
    for (char c : postfix) {
        if (isOperand(c)) {
            stack[++top] = std::string(1, c);
        } else {
            std::string op2 = stack[top--];
            std::string op1 = stack[top--];
            stack[++top] = "(" + op1 + c + op2 + ")";
        }
    }
    return stack[top];
}

TEST(Postfix2Infix, Simple) {
    EXPECT_EQ(Postfix2Infix("ab+"), "(a+b)");
}

TEST(Postfix2Infix, Complex) {
    EXPECT_EQ(Postfix2Infix("ab+cd-*"), "((a+b)*(c-d))");
}

TEST(Postfix2Infix, SingleOperand) {
    EXPECT_EQ(Postfix2Infix("a"), "a");
}

TEST(Postfix2Infix, IsOperand) {
    EXPECT_TRUE(isOperand('a'));
    EXPECT_TRUE(isOperand('Z'));
    EXPECT_FALSE(isOperand('+'));
    EXPECT_FALSE(isOperand('1'));
}
