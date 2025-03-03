/**
 * @file PostfixToInfix.hpp
 * @brief Converts postfix (reverse Polish) expressions to fully-parenthesized infix.
 * @details Uses a stack-based algorithm. Operands are pushed as strings; when an
 *          operator is encountered, two operands are popped, wrapped in parentheses
 *          with the operator between them, and the result is pushed back.
 *
 * Example: "ab*c+" -> "((a*b)+c)"
 *
 * Complexity:
 *   Time  : O(n) where n = length of the postfix expression
 *   Space : O(n) for the stack
 */

#pragma once

#include <stack>
#include <stdexcept>
#include <string>

/**
 * @class PostfixToInfix
 * @brief Converter from postfix notation to infix notation.
 */
class PostfixToInfix {
public:
    /**
     * @brief Convert a postfix expression to a fully-parenthesized infix expression.
     * @param postfix The postfix expression string (single-char operands a-z, A-Z).
     * @return The equivalent infix expression with full parenthesization.
     * @throws std::invalid_argument if the expression is malformed.
     *
     * Supported operators: +, -, *, /, ^
     * Operands: single alphabetic characters (a-z, A-Z)
     */
    static std::string Convert(const std::string& postfix) {
        if (postfix.empty()) {
            return "";
        }

        std::stack<std::string> stk;

        for (char ch : postfix) {
            if (IsOperand(ch)) {
                stk.push(std::string(1, ch));
            } else if (IsOperator(ch)) {
                if (stk.size() < 2) {
                    throw std::invalid_argument(
                        "PostfixToInfix: malformed expression (insufficient operands)");
                }
                std::string op1 = stk.top(); stk.pop();
                std::string op2 = stk.top(); stk.pop();
                std::string result = "(" + op2 + ch + op1 + ")";
                stk.push(result);
            } else {
                throw std::invalid_argument(
                    std::string("PostfixToInfix: invalid character '") + ch + "'");
            }
        }

        if (stk.size() != 1) {
            throw std::invalid_argument(
                "PostfixToInfix: malformed expression (leftover operands)");
        }

        return stk.top();
    }

private:
    /** @brief Check if a character is an operand (letter). */
    static bool IsOperand(char ch) noexcept {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
    }

    /** @brief Check if a character is a supported operator. */
    static bool IsOperator(char ch) noexcept {
        return ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^';
    }
};
