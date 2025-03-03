/**
 * @file main.cpp
 * @brief Driver for CompilerModule
 * @details Reads C++ source from stdin or a hardcoded example, compiles it
 *          through the lexer, parser, semantic analyzer, and code generator,
 *          then prints the generated pseudo-assembly.
 */

#include "Compiler.hpp"
#include <iostream>

int main() {
    std::string source = R"(
        int factorial(int n) {
            if (n <= 1) {
                return 1;
            }
            return n;
        }

        int main() {
            int x = 10;
            int y = 20;
            return x;
        }
    )";

    CompilerModule::Compiler compiler(source);

    if (compiler.Compile()) {
        std::cout << "=== Compilation Successful ===" << std::endl;
        std::cout << compiler.GetGeneratedCode() << std::endl;
    } else {
        std::cerr << "Compilation failed." << std::endl;
        return 1;
    }

    return 0;
}
