/**
 * @file PolynomialCalculator.cpp
 * @brief Implementation of polynomial parsing utilities
 */

#include "PolynomialCalculator.hpp"

namespace PolynomialCalculator {

std::string RemoveSpaces(const std::string& s) {
    std::string result = s;
    result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
    return result;
}

bool HasNumber(const std::string& s) {
    for (char c : s) {
        if (std::isdigit(static_cast<unsigned char>(c))) return true;
    }
    return false;
}

bool ContainsAsterisk(const std::string& s) {
    return s.find('*') != std::string::npos;
}

std::vector<std::string> SplitPolynomial(const std::string& poly) {
    std::vector<std::string> result;
    std::string current;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        if ((poly[i] == '+' || poly[i] == '-') && i > 0) {
            if (!current.empty()) result.push_back(current);
            current = std::string(1, poly[i]);
        } else {
            current += poly[i];
        }
    }
    if (!current.empty()) result.push_back(current);
    return result;
}

} // namespace PolynomialCalculator
