/**
 * @file test_LinearSvm.cpp
 * @brief Google Test suite for N-dimensional LinearSvm.
 *
 * Each test prints an educational section header explaining *why*
 * the test matters for understanding SVM behaviour.
 */

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "LinearSvm.hpp"

namespace {

// -----------------------------------------------------------------------
// Educational section printer
// -----------------------------------------------------------------------
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

// -----------------------------------------------------------------------
// Convenience alias
// -----------------------------------------------------------------------
using Svm = LinearSvm<double>;

// -----------------------------------------------------------------------
// Test: Linearly Separable Classification (2D)
// -----------------------------------------------------------------------
TEST(LinearSvmTest, LinearlySeparableClassification) {
    PrintSection("Linearly Separable Classification",
                 "An SVM trained on well-separated clusters must achieve "
                 "100% accuracy on the training set and correctly classify "
                 "new points that lie deep within each cluster.");

    std::vector<Svm::DataPoint> data = {
        {{0.0, 0.0}, -1}, {{1.0, 0.0}, -1}, {{0.0, 1.0}, -1},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}, {{10.0, 11.0}, 1}
    };

    Svm svm;
    svm.Train(data, /*learning_rate=*/0.01, /*epochs=*/2000,
              /*regularization_C=*/1.0);

    // Every training point should be classified correctly
    for (const auto& dp : data) {
        EXPECT_EQ(svm.Predict(dp.features), dp.label)
            << "Misclassified a training point";
    }

    // Unseen points well inside each cluster
    EXPECT_EQ(svm.Predict({0.5, 0.5}), -1);
    EXPECT_EQ(svm.Predict({10.5, 10.5}), 1);

    std::cout << "Weights: ";
    for (auto w : svm.GetWeights()) std::cout << w << " ";
    std::cout << " bias=" << svm.GetBias() << "\n";
}

// -----------------------------------------------------------------------
// Test: Decision Function Signs
// -----------------------------------------------------------------------
TEST(LinearSvmTest, DecisionFunctionSigns) {
    PrintSection("Decision Function Signs",
                 "The raw decision value w.x + b must be negative for "
                 "class -1 points and positive for class +1 points.  "
                 "This verifies the geometric meaning of the hyperplane.");

    std::vector<Svm::DataPoint> data = {
        {{-1.0, -1.0}, -1}, {{-2.0, -1.0}, -1},
        {{ 1.0,  1.0},  1}, {{ 2.0,  1.0},  1}
    };

    Svm svm;
    svm.Train(data, 0.01, 2000, 1.0);

    EXPECT_LT(svm.DecisionFunction({-1.0, -1.0}), 0.0);
    EXPECT_GT(svm.DecisionFunction({ 1.0,  1.0}), 0.0);
}

// -----------------------------------------------------------------------
// Test: Margin Maximization
// -----------------------------------------------------------------------
TEST(LinearSvmTest, MarginMaximization) {
    PrintSection("Margin Maximization",
                 "SVM should find a hyperplane roughly equidistant from "
                 "the two classes.  We verify that the absolute decision "
                 "values for symmetric support-vector-like points are "
                 "approximately equal, confirming the margin is centred.");

    // Symmetric clusters centred at -5 and +5 on each axis
    std::vector<Svm::DataPoint> data = {
        {{-5.0, -5.0}, -1}, {{-4.0, -5.0}, -1}, {{-5.0, -4.0}, -1},
        {{ 5.0,  5.0},  1}, {{ 4.0,  5.0},  1}, {{ 5.0,  4.0},  1}
    };

    Svm svm;
    svm.Train(data, 0.005, 3000, 10.0);

    double d_neg = svm.DecisionFunction({-5.0, -5.0});
    double d_pos = svm.DecisionFunction({ 5.0,  5.0});

    // Both should be on the correct side
    EXPECT_LT(d_neg, 0.0);
    EXPECT_GT(d_pos, 0.0);

    // Magnitudes should be roughly similar (margin is centred)
    double ratio = std::abs(d_neg) / std::abs(d_pos);
    EXPECT_GT(ratio, 0.3) << "Margin is not reasonably centred";
    EXPECT_LT(ratio, 3.0) << "Margin is not reasonably centred";

    std::cout << "d(-5,-5)=" << d_neg << "  d(5,5)=" << d_pos
              << "  ratio=" << ratio << "\n";
}

// -----------------------------------------------------------------------
// Test: High-Dimensional Features (10D)
// -----------------------------------------------------------------------
TEST(LinearSvmTest, HighDimensionalFeatures) {
    PrintSection("High-Dimensional Features (10D)",
                 "SVMs must work in arbitrary dimensions.  Here we "
                 "construct a 10-D linearly-separable dataset where "
                 "class +1 points have all features > 0 and class -1 "
                 "points have all features < 0.");

    constexpr int kDim = 10;
    constexpr int kPointsPerClass = 20;

    std::vector<Svm::DataPoint> data;
    data.reserve(2 * kPointsPerClass);

    // Class -1: features in [-3, -1], class +1: features in [1, 3]
    for (int i = 0; i < kPointsPerClass; ++i) {
        Svm::Vector neg(kDim), pos(kDim);
        for (int d = 0; d < kDim; ++d) {
            // Deterministic spread: -3 + 2*(i mod kPointsPerClass)/kPointsPerClass
            double frac = static_cast<double>(i) / kPointsPerClass;
            neg[d] = -3.0 + 2.0 * frac;  // range [-3, -1)
            pos[d] =  1.0 + 2.0 * frac;  // range [ 1,  3)
        }
        data.push_back({neg, -1});
        data.push_back({pos,  1});
    }

    Svm svm;
    svm.Train(data, 0.001, 3000, 5.0);

    // Check training accuracy
    int correct = 0;
    for (const auto& dp : data) {
        if (svm.Predict(dp.features) == dp.label) ++correct;
    }
    double accuracy = static_cast<double>(correct) / data.size();
    EXPECT_GE(accuracy, 0.95)
        << "10D training accuracy should be >= 95%, got "
        << accuracy * 100.0 << "%";

    // Weight vector should have kDim elements
    EXPECT_EQ(svm.GetWeights().size(), static_cast<std::size_t>(kDim));

    std::cout << "10D accuracy: " << accuracy * 100.0 << "%  ("
              << correct << "/" << data.size() << ")\n";
}

// -----------------------------------------------------------------------
// Test: Regularization Effect
// -----------------------------------------------------------------------
TEST(LinearSvmTest, RegularizationEffect) {
    PrintSection("Regularization Effect",
                 "Stronger regularization (smaller C) should produce a "
                 "weight vector with smaller L2 norm.  This confirms that "
                 "the regularization term is working correctly.");

    std::vector<Svm::DataPoint> data = {
        {{-3.0, -3.0}, -1}, {{-2.0, -3.0}, -1}, {{-3.0, -2.0}, -1},
        {{ 3.0,  3.0},  1}, {{ 2.0,  3.0},  1}, {{ 3.0,  2.0},  1}
    };

    auto WeightNorm = [](const Svm& model) {
        double norm_sq = 0.0;
        for (double w : model.GetWeights()) norm_sq += w * w;
        return std::sqrt(norm_sq);
    };

    // High C => weak regularization => larger weights
    Svm svm_high_c;
    svm_high_c.Train(data, 0.005, 3000, /*C=*/100.0);
    double norm_high = WeightNorm(svm_high_c);

    // Low C => strong regularization => smaller weights
    Svm svm_low_c;
    svm_low_c.Train(data, 0.005, 3000, /*C=*/0.1);
    double norm_low = WeightNorm(svm_low_c);

    EXPECT_LT(norm_low, norm_high)
        << "Stronger regularization (lower C) should yield smaller weight norm. "
        << "norm(C=0.1)=" << norm_low << ", norm(C=100)=" << norm_high;

    std::cout << "||w|| with C=100: " << norm_high << "\n";
    std::cout << "||w|| with C=0.1: " << norm_low << "\n";
}

// -----------------------------------------------------------------------
// Test: Float Specialization
// -----------------------------------------------------------------------
TEST(LinearSvmTest, FloatSpecialization) {
    PrintSection("Float Specialization",
                 "The template must compile and work with float, not just "
                 "double.  This confirms the std::floating_point concept "
                 "constraint is satisfied for single precision.");

    using SvmF = LinearSvm<float>;

    std::vector<SvmF::DataPoint> data = {
        {{0.0f, 0.0f}, -1}, {{1.0f, 0.0f}, -1},
        {{5.0f, 5.0f},  1}, {{6.0f, 5.0f},  1}
    };

    SvmF svm;
    svm.Train(data, 0.01f, 2000, 1.0f);

    EXPECT_EQ(svm.Predict({0.5f, 0.0f}), -1);
    EXPECT_EQ(svm.Predict({5.5f, 5.0f}),  1);

    std::cout << "float bias=" << svm.GetBias() << "\n";
}

}  // namespace
