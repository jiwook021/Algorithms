/**
 * @file test_NaiveBayesClassifier.cpp
 * @brief Google Test suite for NaiveBayesClassifier.
 * @details Verifies Gaussian classification, multi-class support (3 classes),
 *          log-probability stability, and high-dimensional (8D) classification.
 */

#include "NaiveBayesClassifier.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

constexpr double kTol = 1e-6;

using NBC = NaiveBayesClassifier<double>;
using DP  = NBC::DataPoint;

// ---------------------------------------------------------------------------
// 1. Gaussian Binary Classification
// ---------------------------------------------------------------------------
class GaussianClassificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("Gaussian Binary Classification",
                     "Two well-separated 2D clusters must be classified "
                     "correctly by the Gaussian likelihood model.");
    }
};

TEST_F(GaussianClassificationTest, WellSeparatedClusters) {
    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0},
        {{0.0, 1.0}, 0}, {{1.0, 1.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1},
        {{10.0, 11.0}, 1}, {{11.0, 11.0}, 1},
    };
    NBC nbc;
    nbc.Fit(data);
    EXPECT_EQ(nbc.Predict({0.5, 0.5}), 0);
    EXPECT_EQ(nbc.Predict({10.5, 10.5}), 1);
}

TEST_F(GaussianClassificationTest, PriorProbabilities) {
    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}, {{10.0, 11.0}, 1},
    };
    NBC nbc;
    nbc.Fit(data);

    // log_prior(class 0) = log(2/5), log_prior(class 1) = log(3/5)
    const auto& stats = nbc.GetClassStats();
    EXPECT_NEAR(std::exp(stats.at(0).log_prior), 2.0 / 5.0, kTol);
    EXPECT_NEAR(std::exp(stats.at(1).log_prior), 3.0 / 5.0, kTol);
}

TEST_F(GaussianClassificationTest, LearnedMeans) {
    std::vector<DP> data = {
        {{2.0, 4.0}, 0}, {{4.0, 6.0}, 0},
        {{10.0, 20.0}, 1}, {{12.0, 22.0}, 1},
    };
    NBC nbc;
    nbc.Fit(data);

    const auto& stats = nbc.GetClassStats();
    EXPECT_NEAR(stats.at(0).mean[0], 3.0, kTol);
    EXPECT_NEAR(stats.at(0).mean[1], 5.0, kTol);
    EXPECT_NEAR(stats.at(1).mean[0], 11.0, kTol);
    EXPECT_NEAR(stats.at(1).mean[1], 21.0, kTol);
}

// ---------------------------------------------------------------------------
// 2. Multi-Class Support (3 classes)
// ---------------------------------------------------------------------------
class MultiClassTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("Multi-Class Support (K=3)",
                     "Naive Bayes should handle three or more classes by "
                     "computing separate Gaussian statistics per class.");
    }
};

TEST_F(MultiClassTest, ThreeClasses2D) {
    std::vector<DP> data = {
        // Class 0 -- cluster near (0, 0)
        {{0.0, 0.0}, 0}, {{1.0, 1.0}, 0}, {{0.5, 0.5}, 0},
        // Class 1 -- cluster near (10, 0)
        {{10.0, 0.0}, 1}, {{11.0, 1.0}, 1}, {{10.5, 0.5}, 1},
        // Class 2 -- cluster near (5, 10)
        {{5.0, 10.0}, 2}, {{6.0, 11.0}, 2}, {{5.5, 10.5}, 2},
    };
    NBC nbc;
    nbc.Fit(data);

    EXPECT_EQ(nbc.Predict({0.2, 0.2}), 0);
    EXPECT_EQ(nbc.Predict({10.2, 0.2}), 1);
    EXPECT_EQ(nbc.Predict({5.2, 10.2}), 2);
}

TEST_F(MultiClassTest, FourClassesSeparated) {
    std::vector<DP> data;
    // Four clusters at corners of a square
    double offsets[4][2] = {{0, 0}, {20, 0}, {0, 20}, {20, 20}};
    for (int cls = 0; cls < 4; ++cls) {
        for (int i = 0; i < 5; ++i) {
            data.push_back({{offsets[cls][0] + i * 0.1,
                             offsets[cls][1] + i * 0.1}, cls});
        }
    }
    NBC nbc;
    nbc.Fit(data);

    EXPECT_EQ(nbc.Predict({0.25, 0.25}), 0);
    EXPECT_EQ(nbc.Predict({20.25, 0.25}), 1);
    EXPECT_EQ(nbc.Predict({0.25, 20.25}), 2);
    EXPECT_EQ(nbc.Predict({20.25, 20.25}), 3);
}

// ---------------------------------------------------------------------------
// 3. Log-Probability Stability
// ---------------------------------------------------------------------------
class LogProbabilityStabilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("Log-Probability Stability",
                     "Log-space computation avoids underflow that would "
                     "occur with raw probability products over many features.");
    }
};

TEST_F(LogProbabilityStabilityTest, LogProbsAreFinite) {
    // 2-class, 2-feature simple data.
    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 1.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 11.0}, 1},
    };
    NBC nbc;
    nbc.Fit(data);

    auto log_probs = nbc.PredictLogProbabilities({5.0, 5.0});
    ASSERT_EQ(log_probs.size(), 2u);
    for (auto lp : log_probs) {
        EXPECT_TRUE(std::isfinite(lp));
    }
}

TEST_F(LogProbabilityStabilityTest, LogProbsAreNegative) {
    // Unnormalized log-posteriors should be negative (log of values < 1).
    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0}, {{0.0, 1.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}, {{10.0, 11.0}, 1},
    };
    NBC nbc;
    nbc.Fit(data);

    auto log_probs = nbc.PredictLogProbabilities({0.5, 0.5});
    for (auto lp : log_probs) {
        EXPECT_LT(lp, 0.0);
    }
}

TEST_F(LogProbabilityStabilityTest, BestClassMatchesPredict) {
    std::vector<DP> data = {
        {{1.0, 2.0}, 0}, {{2.0, 3.0}, 0}, {{1.5, 2.5}, 0},
        {{8.0, 9.0}, 1}, {{9.0, 10.0}, 1}, {{8.5, 9.5}, 1},
        {{15.0, 1.0}, 2}, {{16.0, 2.0}, 2}, {{15.5, 1.5}, 2},
    };
    NBC nbc;
    nbc.Fit(data);

    NBC::Vector query = {8.5, 9.5};
    auto log_probs = nbc.PredictLogProbabilities(query);
    ASSERT_EQ(log_probs.size(), 3u);

    // Best log-probability should correspond to predicted class.
    int best_idx = 0;
    for (std::size_t i = 1; i < log_probs.size(); ++i) {
        if (log_probs[i] > log_probs[best_idx]) {
            best_idx = static_cast<int>(i);
        }
    }
    EXPECT_EQ(best_idx, nbc.Predict(query));
}

TEST_F(LogProbabilityStabilityTest, ExtremeFeatureValues) {
    // Even with features far from all class means, log-probs stay finite.
    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 1.0}, 0},
        {{100.0, 100.0}, 1}, {{101.0, 101.0}, 1},
    };
    NBC nbc;
    nbc.Fit(data);

    auto log_probs = nbc.PredictLogProbabilities({1e6, 1e6});
    for (auto lp : log_probs) {
        EXPECT_TRUE(std::isfinite(lp));
    }
}

// ---------------------------------------------------------------------------
// 4. High-Dimensional Classification (8D)
// ---------------------------------------------------------------------------
class HighDimensionalTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("High-Dimensional Classification (8D)",
                     "With many features the product of Gaussian PDFs can "
                     "underflow; log-space arithmetic keeps values tractable.");
    }
};

TEST_F(HighDimensionalTest, EightFeaturesTwoClasses) {
    // Class 0: features centered at 0, class 1: features centered at 10.
    std::vector<DP> data;
    for (int i = 0; i < 10; ++i) {
        NBC::Vector f0(8, static_cast<double>(i) * 0.1);
        NBC::Vector f1(8, 10.0 + static_cast<double>(i) * 0.1);
        data.push_back({f0, 0});
        data.push_back({f1, 1});
    }

    NBC nbc;
    nbc.Fit(data);

    NBC::Vector near_zero(8, 0.5);
    NBC::Vector near_ten(8, 10.5);

    EXPECT_EQ(nbc.Predict(near_zero), 0);
    EXPECT_EQ(nbc.Predict(near_ten), 1);
}

TEST_F(HighDimensionalTest, EightFeaturesThreeClasses) {
    std::vector<DP> data;
    const double centers[3] = {0.0, 20.0, 40.0};
    for (int cls = 0; cls < 3; ++cls) {
        for (int i = 0; i < 8; ++i) {
            NBC::Vector f(8, centers[cls] + static_cast<double>(i) * 0.1);
            data.push_back({f, cls});
        }
    }

    NBC nbc;
    nbc.Fit(data);

    EXPECT_EQ(nbc.Predict(NBC::Vector(8, 0.5)), 0);
    EXPECT_EQ(nbc.Predict(NBC::Vector(8, 20.5)), 1);
    EXPECT_EQ(nbc.Predict(NBC::Vector(8, 40.5)), 2);
}

TEST_F(HighDimensionalTest, LogProbsFiniteIn8D) {
    std::vector<DP> data;
    for (int i = 0; i < 10; ++i) {
        NBC::Vector f0(8, static_cast<double>(i) * 0.1);
        NBC::Vector f1(8, 10.0 + static_cast<double>(i) * 0.1);
        data.push_back({f0, 0});
        data.push_back({f1, 1});
    }

    NBC nbc;
    nbc.Fit(data);

    auto log_probs = nbc.PredictLogProbabilities(NBC::Vector(8, 5.0));
    ASSERT_EQ(log_probs.size(), 2u);
    for (auto lp : log_probs) {
        EXPECT_TRUE(std::isfinite(lp));
    }
}

// ---------------------------------------------------------------------------
// 5. Variance Floor
// ---------------------------------------------------------------------------
class VarianceFloorTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("Variance Floor",
                     "When all training samples for a feature are identical "
                     "the sample variance is zero. The floor prevents NaN.");
    }
};

TEST_F(VarianceFloorTest, IdenticalFeatureValues) {
    // All x_0 values are 5.0 for class 0  =>  variance would be 0.
    std::vector<DP> data = {
        {{5.0, 1.0}, 0}, {{5.0, 2.0}, 0}, {{5.0, 3.0}, 0},
        {{15.0, 1.0}, 1}, {{15.0, 2.0}, 1}, {{15.0, 3.0}, 1},
    };
    NBC nbc;
    nbc.Fit(data);

    // Prediction must still work (no NaN / Inf).
    int pred = nbc.Predict({5.0, 2.0});
    EXPECT_EQ(pred, 0);

    auto log_probs = nbc.PredictLogProbabilities({5.0, 2.0});
    for (auto lp : log_probs) {
        EXPECT_TRUE(std::isfinite(lp));
    }
}

// ---------------------------------------------------------------------------
// 6. Float instantiation
// ---------------------------------------------------------------------------
class FloatInstantiationTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("Float Instantiation",
                     "The template should compile and produce correct "
                     "predictions when instantiated with float.");
    }
};

TEST_F(FloatInstantiationTest, BasicFloat) {
    using NBCf = NaiveBayesClassifier<float>;
    std::vector<NBCf::DataPoint> data = {
        {{0.0f, 0.0f}, 0}, {{1.0f, 1.0f}, 0},
        {{10.0f, 10.0f}, 1}, {{11.0f, 11.0f}, 1},
    };
    NBCf nbc;
    nbc.Fit(data);
    EXPECT_EQ(nbc.Predict({0.5f, 0.5f}), 0);
    EXPECT_EQ(nbc.Predict({10.5f, 10.5f}), 1);
}

}  // namespace
