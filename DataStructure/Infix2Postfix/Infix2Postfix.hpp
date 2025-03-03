/**
 * @file infix2postfix.hpp
 * @brief Infix-to-postfix converter using the Shunting-Yard algorithm.
 *
 * Converts infix mathematical expressions (e.g. "A*(B+C)")
 * to postfix / Reverse Polish Notation (e.g. "ABC+*").
 * Supports: +, -, *, /, ^, parentheses, multi-char operands.
 *
 * Complexity: O(n) time and space, where n = input length.
 */

#pragma once

#include <cctype>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// =============================================================================
// Shunting-Yard — infix → postfix conversion
// =============================================================================

// namespace groups related functions together to avoid name collisions.
// Without it, to_postfix() and evaluate_postfix() would be global names that
// could clash with other code.  With the namespace, they are accessed as:
//   shunting_yard::to_postfix("A+B")
//   shunting_yard::evaluate_postfix("AB+")
// Or you can write  `using shunting_yard::to_postfix;`  at the top of your
// test file to use the short name directly.
namespace ShuntingYard {

/// Operator precedence (higher = binds tighter).
inline int Precedence(char Op) {
    switch (Op) {
        case '+': case '-': return 1;
        case '*': case '/': return 2;
        case '^':           return 3;
        default:            return 0;
    }
}

/// True if @p op is right-associative (only ^ in standard math).
inline bool IsRightAssociative(char Op) {
    return Op == '^';
}

inline bool IsOperator(char Ch) {
    return Ch == '+' || Ch == '-' || Ch == '*' || Ch == '/' || Ch == '^';
}

/**
 * @brief Convert an infix expression string to postfix (RPN).
 *
 * Operands are single letters (A-Z, a-z) or digits.  Operators are
 * +, -, *, /, ^.  Parentheses control grouping.  Whitespace is ignored.
 *
 * @param infix  The infix expression.
 * @return       The equivalent postfix string.
 * @throws std::invalid_argument on mismatched parentheses.
 */
inline std::string ToPostfix(std::string_view Infix) {
    std::string Output;
    std::vector<char> OpStack;

    Output.reserve(Infix.size());

    for (std::size_t i = 0; i < Infix.size(); ++i) {
        char Ch = Infix[i];

        if (std::isspace(static_cast<unsigned char>(Ch)))
            continue;

        // Operand — letter or digit
        if (std::isalnum(static_cast<unsigned char>(Ch))) {
            Output += Ch;
            continue;
        }

        // Left parenthesis
        if (Ch == '(') {
            OpStack.push_back(Ch);
            continue;
        }

        // Right parenthesis — pop until matching '('
        if (Ch == ')') {
            while (!OpStack.empty() && OpStack.back() != '(') {
                Output += OpStack.back();
                OpStack.pop_back();
            }
            if (OpStack.empty())
                throw std::invalid_argument("Mismatched parentheses: extra ')'");
            OpStack.pop_back();  // discard '('
            continue;
        }

        // Operator
        if (IsOperator(Ch)) {
            while (!OpStack.empty() && OpStack.back() != '(' &&
                   (Precedence(OpStack.back()) > Precedence(Ch) ||
                    (Precedence(OpStack.back()) == Precedence(Ch) &&
                     !IsRightAssociative(Ch)))) {
                Output += OpStack.back();
                OpStack.pop_back();
            }
            OpStack.push_back(Ch);
            continue;
        }

        throw std::invalid_argument(
            std::string("Unexpected character: '") + Ch + "'");
    }

    // Drain remaining operators
    while (!OpStack.empty()) {
        if (OpStack.back() == '(')
            throw std::invalid_argument("Mismatched parentheses: extra '('");
        Output += OpStack.back();
        OpStack.pop_back();
    }

    return Output;
}

/**
 * @brief Evaluate a postfix expression with single-digit operands.
 *
 * Useful for verifying conversion correctness.
 *
 * @param postfix  A postfix expression where operands are '0'–'9'.
 * @return         The integer result.
 */
inline int EvaluatePostfix(std::string_view Postfix) {
    std::vector<int> St;

    for (char Ch : Postfix) {
        if (std::isdigit(static_cast<unsigned char>(Ch))) {
            St.push_back(Ch - '0');
        } else if (IsOperator(Ch)) {
            if (St.size() < 2)
                throw std::invalid_argument("Invalid postfix expression");
            int b = St.back(); St.pop_back();
            int a = St.back(); St.pop_back();
            switch (Ch) {
                case '+': St.push_back(a + b); break;
                case '-': St.push_back(a - b); break;
                case '*': St.push_back(a * b); break;
                case '/':
                    if (b == 0) throw std::invalid_argument("Division by zero");
                    St.push_back(a / b);
                    break;
                case '^': {
                    int r = 1;
                    for (int i = 0; i < b; ++i) r *= a;
                    St.push_back(r);
                    break;
                }
                default: break;
            }
        }
    }

    if (St.size() != 1)
        throw std::invalid_argument("Invalid postfix expression");
    return St.back();
}

} // namespace shunting_yard
