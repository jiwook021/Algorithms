/**
 * @file test_AddPolynomial.cpp
 * @brief Unit tests for AddPolynomial
 */

#include "AddPolynomial.hpp"
#include <gtest/gtest.h>

using namespace AddPolynomial;

TEST(Polynomial, AddSameVariables) {
    Polynomial p1;
    p1.AddTerm(3.0, {{1, 2}});

    Polynomial p2;
    p2.AddTerm(2.0, {{1, 2}});

    auto result = p1 + p2;
    EXPECT_EQ(result.TermCount(), 1u);
    EXPECT_DOUBLE_EQ(result.GetTerms().front().Coeff, 5.0);
}

TEST(Polynomial, AddDifferentVariables) {
    Polynomial p1;
    p1.AddTerm(3.0, {{1, 2}});

    Polynomial p2;
    p2.AddTerm(2.0, {{2, 1}});

    auto result = p1 + p2;
    EXPECT_EQ(result.TermCount(), 2u);
}

TEST(Polynomial, EmptyPolynomials) {
    Polynomial p1, p2;
    auto result = p1 + p2;
    EXPECT_EQ(result.TermCount(), 0u);
}

TEST(Polynomial, AddToEmpty) {
    Polynomial p1;
    Polynomial p2;
    p2.AddTerm(5.0, {{1, 1}});

    auto result = p1 + p2;
    EXPECT_EQ(result.TermCount(), 1u);
    EXPECT_DOUBLE_EQ(result.GetTerms().front().Coeff, 5.0);
}

TEST(Polynomial, MultipleTermsCombine) {
    Polynomial p1;
    p1.AddTerm(1.0, {{1, 1}});
    p1.AddTerm(2.0, {{1, 2}});

    Polynomial p2;
    p2.AddTerm(3.0, {{1, 1}});
    p2.AddTerm(4.0, {{1, 3}});

    auto result = p1 + p2;
    EXPECT_EQ(result.TermCount(), 3u);

    // x^1 term should be combined: 1 + 3 = 4
    for (const auto& t : result.GetTerms()) {
        if (t.Vars.front().Exp == 1) {
            EXPECT_DOUBLE_EQ(t.Coeff, 4.0);
        }
    }
}

TEST(Polynomial, AddTermCombinesDuplicates) {
    Polynomial p;
    p.AddTerm(3.0, {{1, 2}});
    p.AddTerm(7.0, {{1, 2}});
    EXPECT_EQ(p.TermCount(), 1u);
    EXPECT_DOUBLE_EQ(p.GetTerms().front().Coeff, 10.0);
}
