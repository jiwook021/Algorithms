/**
 * @file test_LinearDiscriminantAnalysis.cpp
 * @brief Google Test suite for N-dimensional Fisher's LDA.
 *
 * Covers: class separability maximization, projection discriminability,
 *         high-dimensional (5D->1D) projection, training accuracy,
 *         weight normalization, and edge cases.
 */

#include <gtest/gtest.h>

#include "LinearDiscriminantAnalysis.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// PrintSection helper -- educational banners before each test
// ---------------------------------------------------------------------------

namespace {
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}
}  // namespace

// ===========================================================================
// Type alias for convenience
// ===========================================================================

using LDA = LinearDiscriminantAnalysis<double>;
using DP  = LDA::DataPoint;

// ===========================================================================
// MaximizesClassSeparability
// ===========================================================================

TEST(LinearDiscriminantAnalysisTest, MaximizesClassSeparability) {
    PrintSection("MaximizesClassSeparability",
                 "Fisher's criterion maximizes the ratio of between-class "
                 "to within-class variance.  After projection the class "
                 "means should be well separated relative to their spread.");

    // Two well-separated 2D clusters
    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0}, {{0.0, 1.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}, {{10.0, 11.0}, 1}
    };

    LDA lda;
    lda.Fit(data);

    // Project each class and compute means
    double sum0 = 0, sum1 = 0;
    int n0 = 0, n1 = 0;
    for (const auto& dp : data) {
        double proj = lda.Project(dp.features)[0];
        if (dp.label == 0) { sum0 += proj; ++n0; }
        else               { sum1 += proj; ++n1; }
    }
    double mean0 = sum0 / n0;
    double mean1 = sum1 / n1;

    // The projected means should be clearly separated
    double separation = std::fabs(mean1 - mean0);
    std::cout << "  Projected mean class 0: " << mean0 << "\n";
    std::cout << "  Projected mean class 1: " << mean1 << "\n";
    std::cout << "  Separation: " << separation << "\n";

    EXPECT_GT(separation, 1.0)
        << "Projected class means should be well separated";
}

// ===========================================================================
// ProjectionPreservesDiscriminability
// ===========================================================================

TEST(LinearDiscriminantAnalysisTest, ProjectionPreservesDiscriminability) {
    PrintSection("ProjectionPreservesDiscriminability",
                 "All class 0 projections should be below all class 1 "
                 "projections when the data is linearly separable.");

    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 1.0}, 0},
        {{5.0, 5.0}, 1}, {{6.0, 6.0}, 1}
    };

    LDA lda;
    lda.Fit(data);

    std::vector<double> proj0, proj1;
    for (const auto& dp : data) {
        double p = lda.Project(dp.features)[0];
        if (dp.label == 0) proj0.push_back(p);
        else               proj1.push_back(p);
    }

    double max0 = *std::max_element(proj0.begin(), proj0.end());
    double min1 = *std::min_element(proj1.begin(), proj1.end());

    std::cout << "  Max projection class 0: " << max0 << "\n";
    std::cout << "  Min projection class 1: " << min1 << "\n";

    EXPECT_LT(max0, min1)
        << "Class 0 projections should all be below class 1 projections";
}

// ===========================================================================
// HighDimensionalProjection  (5D -> 1D)
// ===========================================================================

TEST(LinearDiscriminantAnalysisTest, HighDimensionalProjection) {
    PrintSection("HighDimensionalProjection",
                 "Fisher's LDA generalizes to N dimensions. Here we project "
                 "5D data to 1D and verify classification accuracy.");

    // Class 0: features near 1.0,  Class 1: features near 10.0
    std::vector<DP> data = {
        {{1.0, 1.1, 0.9, 1.2, 0.8}, 0},
        {{1.1, 0.9, 1.0, 0.8, 1.2}, 0},
        {{0.9, 1.0, 1.1, 1.0, 1.0}, 0},
        {{0.8, 1.2, 0.9, 1.1, 0.9}, 0},
        {{10.0, 10.1, 9.9, 10.2, 9.8}, 1},
        {{10.1, 9.9, 10.0, 9.8, 10.2}, 1},
        {{9.9, 10.0, 10.1, 10.0, 10.0}, 1},
        {{9.8, 10.2, 9.9, 10.1, 9.9}, 1}
    };

    LDA lda;
    lda.Fit(data);

    const auto& w = lda.GetProjectionVector();
    std::cout << "  Projection vector (5D): [";
    for (size_t i = 0; i < w.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << w[i];
    }
    std::cout << "]\n";
    std::cout << "  Threshold: " << lda.GetThreshold() << "\n";

    EXPECT_EQ(w.size(), 5u)
        << "Projection vector should have 5 components";

    // Every point should be classified correctly
    int correct = 0;
    for (const auto& dp : data) {
        if (lda.Predict(dp.features) == dp.label) ++correct;
    }
    std::cout << "  Accuracy: " << correct << "/" << data.size() << "\n";

    EXPECT_EQ(correct, static_cast<int>(data.size()))
        << "100% accuracy expected on well-separated 5D data";
}

// ===========================================================================
// TrainingAccuracy2D
// ===========================================================================

TEST(LinearDiscriminantAnalysisTest, TrainingAccuracy2D) {
    PrintSection("TrainingAccuracy2D",
                 "On linearly separable 2D data, training accuracy should "
                 "be 100%.  This confirms the decision boundary is correct.");

    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0}, {{0.0, 1.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}, {{10.0, 11.0}, 1}
    };

    LDA lda;
    lda.Fit(data);

    int correct = 0;
    for (const auto& dp : data) {
        int pred = lda.Predict(dp.features);
        std::cout << "  features=[" << dp.features[0] << ", "
                  << dp.features[1] << "]  label=" << dp.label
                  << "  pred=" << pred << "\n";
        if (pred == dp.label) ++correct;
    }
    EXPECT_EQ(correct, static_cast<int>(data.size()));
}

// ===========================================================================
// WeightVectorIsNormalized
// ===========================================================================

TEST(LinearDiscriminantAnalysisTest, WeightVectorIsNormalized) {
    PrintSection("WeightVectorIsNormalized",
                 "The projection vector w should be normalized to unit length "
                 "so that the projected values are directly comparable.");

    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}
    };

    LDA lda;
    lda.Fit(data);

    const auto& w = lda.GetProjectionVector();
    double norm = 0;
    for (double wi : w) {
        norm += wi * wi;
    }
    norm = std::sqrt(norm);

    std::cout << "  ||w|| = " << norm << "\n";
    EXPECT_NEAR(norm, 1.0, 1e-6)
        << "Weight vector should be unit length";
}

// ===========================================================================
// ThresholdBetweenClassMeans
// ===========================================================================

TEST(LinearDiscriminantAnalysisTest, ThresholdBetweenClassMeans) {
    PrintSection("ThresholdBetweenClassMeans",
                 "The decision threshold should lie between the projected "
                 "means of the two classes.");

    std::vector<DP> data = {
        {{0.0, 0.0}, 0}, {{1.0, 1.0}, 0}, {{2.0, 2.0}, 0},
        {{8.0, 8.0}, 1}, {{9.0, 9.0}, 1}, {{10.0, 10.0}, 1}
    };

    LDA lda;
    lda.Fit(data);

    // Compute projected means
    double sum0 = 0, sum1 = 0;
    int n0 = 0, n1 = 0;
    for (const auto& dp : data) {
        double p = lda.Project(dp.features)[0];
        if (dp.label == 0) { sum0 += p; ++n0; }
        else               { sum1 += p; ++n1; }
    }
    double pm0 = sum0 / n0;
    double pm1 = sum1 / n1;
    double low  = std::min(pm0, pm1);
    double high = std::max(pm0, pm1);
    double thr  = lda.GetThreshold();

    std::cout << "  Projected mean 0: " << pm0 << "\n";
    std::cout << "  Projected mean 1: " << pm1 << "\n";
    std::cout << "  Threshold: " << thr << "\n";

    EXPECT_GE(thr, low - 1e-9);
    EXPECT_LE(thr, high + 1e-9);
}

// ===========================================================================
// FloatInstantiation
// ===========================================================================

TEST(LinearDiscriminantAnalysisTest, FloatInstantiation) {
    PrintSection("FloatInstantiation",
                 "The template should compile and work with float precision.");

    using LDAF = LinearDiscriminantAnalysis<float>;
    using DPF  = LDAF::DataPoint;

    std::vector<DPF> data = {
        {{0.0f, 0.0f}, 0}, {{1.0f, 0.0f}, 0},
        {{10.0f, 10.0f}, 1}, {{11.0f, 10.0f}, 1}
    };

    LDAF lda;
    lda.Fit(data);

    EXPECT_EQ(lda.Predict({0.5f, 0.0f}), 0);
    EXPECT_EQ(lda.Predict({10.5f, 10.0f}), 1);

    float norm = 0;
    for (float wi : lda.GetProjectionVector()) {
        norm += wi * wi;
    }
    norm = std::sqrt(norm);
    std::cout << "  float ||w|| = " << norm << "\n";

    EXPECT_NEAR(norm, 1.0f, 1e-4f);
}

// ===========================================================================
// ThreeDimensionalProjection
// ===========================================================================

TEST(LinearDiscriminantAnalysisTest, ThreeDimensionalProjection) {
    PrintSection("ThreeDimensionalProjection",
                 "Verify LDA handles 3D data where classes differ mostly "
                 "along one axis -- the projection vector should weight "
                 "that axis more heavily.");

    // Classes separated primarily along dimension 0
    std::vector<DP> data = {
        {{0.0, 5.0, 5.0}, 0}, {{0.5, 4.5, 5.5}, 0}, {{-0.5, 5.5, 4.5}, 0},
        {{10.0, 5.0, 5.0}, 1}, {{10.5, 4.5, 5.5}, 1}, {{9.5, 5.5, 4.5}, 1}
    };

    LDA lda;
    lda.Fit(data);

    const auto& w = lda.GetProjectionVector();
    std::cout << "  w = [" << w[0] << ", " << w[1] << ", " << w[2] << "]\n";

    // The first component should dominate since that is the separating axis
    EXPECT_GT(std::fabs(w[0]), std::fabs(w[1]))
        << "First component should dominate when separation is along dim 0";
    EXPECT_GT(std::fabs(w[0]), std::fabs(w[2]))
        << "First component should dominate when separation is along dim 0";

    // Classification should still be perfect
    for (const auto& dp : data) {
        EXPECT_EQ(lda.Predict(dp.features), dp.label);
    }
}
