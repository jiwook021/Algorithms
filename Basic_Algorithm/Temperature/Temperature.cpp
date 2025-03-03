/**
 * @file Temperature.cpp
 * @brief Implementation of temperature conversion functions
 */

#include "Temperature.hpp"

namespace Temperature {

double CelsiusToFahrenheit(double celsius) {
    return (9.0 / 5.0) * celsius + 32.0;
}

double FahrenheitToCelsius(double fahrenheit) {
    return (5.0 / 9.0) * (fahrenheit - 32.0);
}

} // namespace Temperature
