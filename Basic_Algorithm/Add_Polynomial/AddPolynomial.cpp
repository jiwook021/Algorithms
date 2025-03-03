/**
 * @file AddPolynomial.cpp
 * @brief Implementation of Polynomial arithmetic
 */

#include "AddPolynomial.hpp"

namespace AddPolynomial {

void Polynomial::AddTerm(double coeff, std::list<Variable> vars) {
    // If a matching term exists, combine coefficients
    for (auto& t : terms_) {
        Term candidate{coeff, vars};
        if (t.SameVariables(candidate)) {
            t.Coeff += coeff;
            return;
        }
    }
    terms_.push_back({coeff, std::move(vars)});
}

Polynomial Polynomial::operator+(const Polynomial& other) const {
    Polynomial result = *this;
    for (const auto& t : other.terms_) {
        bool found = false;
        for (auto& rt : result.terms_) {
            if (rt.SameVariables(t)) {
                rt.Coeff += t.Coeff;
                found = true;
                break;
            }
        }
        if (!found) {
            result.terms_.push_back(t);
        }
    }
    return result;
}

} // namespace AddPolynomial
