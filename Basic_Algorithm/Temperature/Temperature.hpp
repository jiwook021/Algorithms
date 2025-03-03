/**
 * @file Temperature.hpp
 * @brief Temperature conversion functions (Celsius/Fahrenheit)
 */

#pragma once

namespace Temperature {

/**
 * @brief Convert Celsius to Fahrenheit
 * @param celsius Temperature in Celsius
 * @return Temperature in Fahrenheit
 */
double CelsiusToFahrenheit(double celsius);

/**
 * @brief Convert Fahrenheit to Celsius
 * @param fahrenheit Temperature in Fahrenheit
 * @return Temperature in Celsius
 */
double FahrenheitToCelsius(double fahrenheit);

} // namespace Temperature
