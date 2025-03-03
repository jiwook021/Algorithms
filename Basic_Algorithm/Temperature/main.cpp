/**
 * @file main.cpp
 * @brief Driver for Temperature conversion
 */

#include "Temperature.hpp"
#include <iostream>
#include <string>

int main() {
    double value;
    char unit;

    std::cout << "Enter temperature and unit (e.g., 100 C or 212 F): ";
    std::cin >> value >> unit;

    if (unit == 'C' || unit == 'c') {
        std::cout << value << " C = "
                  << Temperature::CelsiusToFahrenheit(value) << " F" << std::endl;
    } else if (unit == 'F' || unit == 'f') {
        std::cout << value << " F = "
                  << Temperature::FahrenheitToCelsius(value) << " C" << std::endl;
    } else {
        std::cerr << "Unknown unit: " << unit << std::endl;
        return 1;
    }

    return 0;
}
