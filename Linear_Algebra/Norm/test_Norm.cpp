/**
 * @file test_Norm.cpp
 * @brief Google Test suite for the Norm module.
 * @details Verifies L1, L2, LInf, and PNorm with educational output
 *          explaining the purpose of each test group.
 */

#include "Norm.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

constexpr double kTol = 1e-9;

// ---------------------------------------------------------------------------
// L1 Norm (Manhattan distance)
// ---------------------------------------------------------------------------
class L1NormTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("L1 Norm (Manhattan)",
                     "Sum of absolute values -- measures taxicab distance.");
    }
};

TEST_F(L1NormTest, BasicVector) {
    // |3| + |-4| + |5| + |-2| = 14
    std::vector<double> v{3.0, -4.0, 5.0, -2.0};
    EXPECT_NEAR(Norm::L1(v), 14.0, kTol);
}

TEST_F(L1NormTest, AllPositive) {
    std::vector<double> v{1.0, 2.0, 3.0};
    EXPECT_NEAR(Norm::L1(v), 6.0, kTol);
}

TEST_F(L1NormTest, SingleElement) {
    std::vector<double> v{-42.0};
    EXPECT_NEAR(Norm::L1(v), 42.0, kTol);
}

TEST_F(L1NormTest, EmptyVector) {
    std::vector<double> v;
    EXPECT_NEAR(Norm::L1(v), 0.0, kTol);
}

TEST_F(L1NormTest, WorksWithArray) {
    std::array<double, 3> a{1.0, -2.0, 3.0};
    EXPECT_NEAR(Norm::L1(a), 6.0, kTol);
}

TEST_F(L1NormTest, WorksWithFloat) {
    std::vector<float> v{1.0f, -2.0f, 3.0f};
    EXPECT_NEAR(Norm::L1(v), 6.0f, 1e-5);
}

// ---------------------------------------------------------------------------
// L2 Norm (Euclidean distance)
// ---------------------------------------------------------------------------
class L2NormTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("L2 Norm (Euclidean)",
                     "Square root of sum of squares -- straight-line distance.");
    }
};

TEST_F(L2NormTest, PythagoreanTriple) {
    // sqrt(9 + 16) = 5
    std::vector<double> v{3.0, 4.0};
    EXPECT_NEAR(Norm::L2(v), 5.0, kTol);
}

TEST_F(L2NormTest, GeneralVector) {
    // sqrt(9 + 16 + 25 + 4) = sqrt(54)
    std::vector<double> v{3.0, -4.0, 5.0, -2.0};
    EXPECT_NEAR(Norm::L2(v), std::sqrt(54.0), kTol);
}

TEST_F(L2NormTest, UnitVector) {
    std::vector<double> v{1.0, 0.0, 0.0};
    EXPECT_NEAR(Norm::L2(v), 1.0, kTol);
}

TEST_F(L2NormTest, SingleElement) {
    std::vector<double> v{-42.0};
    EXPECT_NEAR(Norm::L2(v), 42.0, kTol);
}

TEST_F(L2NormTest, EmptyVector) {
    std::vector<double> v;
    EXPECT_NEAR(Norm::L2(v), 0.0, kTol);
}

TEST_F(L2NormTest, WorksWithArray) {
    std::array<double, 2> a{3.0, 4.0};
    EXPECT_NEAR(Norm::L2(a), 5.0, kTol);
}

// ---------------------------------------------------------------------------
// L-infinity Norm (max absolute value)
// ---------------------------------------------------------------------------
class LInfNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("L-infinity Norm",
                     "Maximum absolute value -- the dominant component.");
    }
};

TEST_F(LInfNormTest, BasicVector) {
    std::vector<double> v{3.0, -7.0, 5.0};
    EXPECT_NEAR(Norm::LInf(v), 7.0, kTol);
}

TEST_F(LInfNormTest, AllEqual) {
    std::vector<double> v{5.0, 5.0, 5.0};
    EXPECT_NEAR(Norm::LInf(v), 5.0, kTol);
}

TEST_F(LInfNormTest, SingleElement) {
    std::vector<double> v{-42.0};
    EXPECT_NEAR(Norm::LInf(v), 42.0, kTol);
}

TEST_F(LInfNormTest, EmptyVector) {
    std::vector<double> v;
    EXPECT_NEAR(Norm::LInf(v), 0.0, kTol);
}

TEST_F(LInfNormTest, WorksWithArray) {
    std::array<double, 3> a{1.0, -9.0, 3.0};
    EXPECT_NEAR(Norm::LInf(a), 9.0, kTol);
}

// ---------------------------------------------------------------------------
// General PNorm and consistency checks
// ---------------------------------------------------------------------------
class PNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("General P-Norm",
                     "Validates PNorm for arbitrary p and consistency "
                     "with L1/L2/LInf shortcuts.");
    }
};

TEST_F(PNormTest, L3Norm) {
    // (|3|^3 + |4|^3)^{1/3} = (27+64)^{1/3} = 91^{1/3}
    std::vector<double> v{3.0, 4.0};
    double expected = std::pow(91.0, 1.0 / 3.0);
    EXPECT_NEAR(Norm::PNorm(v, 3.0), expected, kTol);
}

TEST_F(PNormTest, ConsistentWithL1) {
    std::vector<double> v{3.0, -4.0, 5.0, -2.0};
    EXPECT_NEAR(Norm::PNorm(v, 1.0), Norm::L1(v), kTol);
}

TEST_F(PNormTest, ConsistentWithL2) {
    std::vector<double> v{3.0, -4.0, 5.0, -2.0};
    EXPECT_NEAR(Norm::PNorm(v, 2.0), Norm::L2(v), kTol);
}

TEST_F(PNormTest, ConsistentWithLInf) {
    std::vector<double> v{3.0, -4.0, 5.0, -2.0};
    EXPECT_NEAR(Norm::PNorm(v, std::numeric_limits<double>::infinity()),
                Norm::LInf(v), kTol);
}

TEST_F(PNormTest, EmptyVector) {
    std::vector<double> v;
    EXPECT_NEAR(Norm::PNorm(v, 2.0), 0.0, kTol);
    EXPECT_NEAR(Norm::PNorm(v, std::numeric_limits<double>::infinity()),
                0.0, kTol);
}

TEST_F(PNormTest, InvalidPThrows) {
    std::vector<double> v{1.0};
    EXPECT_THROW(Norm::PNorm(v, 0.5), std::invalid_argument);
    EXPECT_THROW(Norm::PNorm(v, -1.0), std::invalid_argument);
}

TEST_F(PNormTest, HighPApproachesLInf) {
    // As p -> inf, ||x||_p -> ||x||_inf
    std::vector<double> v{1.0, -5.0, 3.0};
    double high_p = Norm::PNorm(v, 100.0);
    double linf   = Norm::LInf(v);
    EXPECT_NEAR(high_p, linf, 1e-3);
}

TEST_F(PNormTest, WorksWithArray) {
    std::array<double, 2> a{3.0, 4.0};
    EXPECT_NEAR(Norm::PNorm(a, 2.0), 5.0, kTol);
}

} // namespace
