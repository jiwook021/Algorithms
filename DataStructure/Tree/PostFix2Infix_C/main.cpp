/**
 * @file main.cpp
 * @brief Driver for PostfixToInfix converter
 */

#include "PostfixToInfix.hpp"
#include <iostream>

int main() {
    std::string postfix1 = "ab*c+";
    std::string postfix2 = "abc/-ad/e-*";

    std::cout << "Postfix: " << postfix1 << "\n";
    std::cout << "Infix:   " << PostfixToInfix::Convert(postfix1) << "\n\n";

    std::cout << "Postfix: " << postfix2 << "\n";
    std::cout << "Infix:   " << PostfixToInfix::Convert(postfix2) << "\n";

    return 0;
}
