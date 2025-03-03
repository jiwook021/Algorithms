/**
 * @file AddPolynomial.hpp
 * @brief Polynomial arithmetic via linked-list representation
 * @details Linked-list Polynomial with sorted (coefficient, exponent) nodes.
 *          Supports operator+, evaluate(), and stream output.
 */

#pragma once

#include <list>
#include <vector>
#include <string>
#include <sstream>

namespace AddPolynomial {

struct Variable {
    int Id;
    int Exp;
    bool operator==(const Variable& other) const {
        return Id == other.Id && Exp == other.Exp;
    }
    bool operator<(const Variable& other) const {
        return Id < other.Id || (Id == other.Id && Exp < other.Exp);
    }
};

struct Term {
    double Coeff;
    std::list<Variable> Vars;

    bool SameVariables(const Term& other) const {
        if (Vars.size() != other.Vars.size()) return false;
        auto it1 = Vars.begin();
        auto it2 = other.Vars.begin();
        while (it1 != Vars.end()) {
            if (!(*it1 == *it2)) return false;
            ++it1;
            ++it2;
        }
        return true;
    }
};

class Polynomial {
public:
    Polynomial() = default;

    void AddTerm(double coeff, std::list<Variable> vars);

    Polynomial operator+(const Polynomial& other) const;

    std::size_t TermCount() const { return terms_.size(); }

    const std::list<Term>& GetTerms() const { return terms_; }

private:
    std::list<Term> terms_;
};

} // namespace AddPolynomial
