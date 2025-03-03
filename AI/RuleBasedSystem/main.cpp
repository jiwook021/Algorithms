/**
 * @file main.cpp
 * @brief Driver for RuleBasedSystem demos
 */

#include "RuleBasedSystem.hpp"
#include <iostream>

int main() {
    try {
        Demo::RunSimpleTest();
        Demo::RunMedicalDiagnosisTest();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
