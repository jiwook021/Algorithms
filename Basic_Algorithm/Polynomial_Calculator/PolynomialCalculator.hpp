/**
 * @file PolynomialCalculator.hpp
 * @brief Polynomial string parsing utilities
 * @details Utilities for parsing polynomial expressions from strings:
 *          remove spaces, detect numbers/asterisks, and split into terms.
 */

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

namespace PolynomialCalculator {

/**
 * @brief Remove all spaces from a string.
 */
std::string RemoveSpaces(const std::string& s);

/**
 * @brief Check if a string contains any digit character.
 */
bool HasNumber(const std::string& s);

/**
 * @brief Check if a string contains an asterisk.
 */
bool ContainsAsterisk(const std::string& s);

/**
 * @brief Split a polynomial string into individual terms.
 *        Splits on '+' and '-' while preserving the sign with each term.
 */
std::vector<std::string> SplitPolynomial(const std::string& poly);

} // namespace PolynomialCalculator
