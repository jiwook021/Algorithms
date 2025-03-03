/**
 * @file main.cpp
 * @brief Driver for Infix2Postfix (Shunting-Yard algorithm)
 */

#include "Infix2Postfix.hpp"
#include <iostream>

int main() {
    using ShuntingYard::ToPostfix;
    using ShuntingYard::EvaluatePostfix;

    const char* expressions[] = {
        "A+B*C",
        "(A+B)*C",
        "A*(B+C*D)+E",
        "3+4*2",
        "(3+4)*2",
        "2^3",
    };

    for (const auto& expr : expressions) {
        auto postfix = ToPostfix(expr);
        std::cout << "Infix:   " << expr << "\n";
        std::cout << "Postfix: " << postfix << "\n\n";
    }

    // Evaluate numeric expressions
    std::cout << "3+4*2 = " << EvaluatePostfix(ToPostfix("3+4*2")) << "\n";
    std::cout << "(3+4)*2 = " << EvaluatePostfix(ToPostfix("(3+4)*2")) << "\n";
    std::cout << "2^3 = " << EvaluatePostfix(ToPostfix("2^3")) << "\n";

    return 0;
}
