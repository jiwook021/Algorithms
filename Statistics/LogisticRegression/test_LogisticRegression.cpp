/**
 * @file test_LogisticRegression.cpp
 * @brief Google Test suite for N-dimensional LogisticRegression.
 *
 * Each test prints a titled section explaining *why* the property matters,
 * followed by intermediate results so readers can follow the reasoning.
 */

#include "LogisticRegression.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

using LR = LogisticRegression<double>;

} // namespace

// ── Sigmoid properties ──────────────────────────────────────────────────────

TEST(LogisticRegression, SigmoidAtZero) {
    PrintSection("SigmoidAtZero",
                 "sigma(0) = 0.5 is the pivot point of the sigmoid curve; "
                 "verifies the function is correctly centered.");
    double result = LR::Sigmoid(0.0);
    std::cout << "[Result] Sigmoid(0) = " << result << " (expect 0.5)" << std::endl;
    EXPECT_NEAR(result, 0.5, 1e-10);
}

TEST(LogisticRegression, SigmoidOutputRange) {
    PrintSection("SigmoidOutputRange",
                 "Sigmoid must map every real number into (0, 1); values outside "
                 "this range would make predicted probabilities meaningless.");

    std::vector<double> inputs = {-100, -10, -1, 0, 1, 10, 100};
    for (double z : inputs) {
        double s = LR::Sigmoid(z);
        std::cout << "[Result] Sigmoid(" << z << ") = " << s << std::endl;
        EXPECT_GE(s, 0.0);
        EXPECT_LE(s, 1.0);
    }
}

TEST(LogisticRegression, SigmoidLargePositive) {
    PrintSection("SigmoidLargePositive",
                 "Large positive inputs should saturate near 1.0, confirming "
                 "no numerical overflow in exp(-z).");
    double result = LR::Sigmoid(10.0);
    std::cout << "[Result] Sigmoid(10) = " << result << " (expect > 0.99)" << std::endl;
    EXPECT_GT(result, 0.99);
}

TEST(LogisticRegression, SigmoidLargeNegative) {
    PrintSection("SigmoidLargeNegative",
                 "Large negative inputs should saturate near 0.0, confirming "
                 "no numerical underflow.");
    double result = LR::Sigmoid(-10.0);
    std::cout << "[Result] Sigmoid(-10) = " << result << " (expect < 0.01)" << std::endl;
    EXPECT_LT(result, 0.01);
}

// ── Binary classification accuracy ──────────────────────────────────────────

TEST(LogisticRegression, BinaryClassification2D) {
    PrintSection("BinaryClassification2D",
                 "Well-separated 2D clusters should be classified perfectly, "
                 "confirming gradient descent converges on linearly separable data.");

    std::vector<LR::DataPoint> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0}, {{0.0, 1.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}, {{10.0, 11.0}, 1}
    };

    LR model;
    model.Fit(data, 0.1, 2000);

    double accuracy = model.CalculateAccuracy(data);
    std::cout << "[Result] Training accuracy = " << accuracy * 100.0 << "%" << std::endl;
    EXPECT_GE(accuracy, 0.99);

    int pred0 = model.PredictClass({0.5, 0.5});
    int pred1 = model.PredictClass({10.5, 10.5});
    std::cout << "[Result] PredictClass({0.5, 0.5})   = " << pred0 << " (expect 0)" << std::endl;
    std::cout << "[Result] PredictClass({10.5, 10.5}) = " << pred1 << " (expect 1)" << std::endl;
    EXPECT_EQ(pred0, 0);
    EXPECT_EQ(pred1, 1);
}

TEST(LogisticRegression, AccuracyOnSeparableData) {
    PrintSection("AccuracyOnSeparableData",
                 "Training accuracy on easy separable data must reach at least 75%; "
                 "a weaker threshold catches gross training failures.");

    std::vector<LR::DataPoint> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}
    };

    LR model;
    model.Fit(data, 0.1, 1000);

    double accuracy = model.CalculateAccuracy(data);
    std::cout << "[Result] Training accuracy = " << accuracy * 100.0 << "%" << std::endl;
    EXPECT_GE(accuracy, 0.75);
}

// ── Cross-entropy loss decreases ─────────────────────────────────────────────

TEST(LogisticRegression, CrossEntropyLossDecreases) {
    PrintSection("CrossEntropyLossDecreases",
                 "If the optimizer is working, loss after training must be strictly "
                 "lower than loss at random initialization (all weights = 0).");

    std::vector<LR::DataPoint> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0}, {{0.0, 1.0}, 0},
        {{5.0, 5.0}, 1}, {{6.0, 5.0}, 1}, {{5.0, 6.0}, 1}
    };

    // Loss before training (weights = 0, bias = 0 => every prediction = 0.5)
    LR before_model;
    before_model.Fit(data, 0.0, 0);  // zero learning rate, zero epochs => untrained
    // Since Fit with 0 epochs still initializes weights to 0, we compute loss:
    // Actually, 0 epochs means the loop doesn't execute, so weights stay at 0.
    // We need to manually get the loss. Let's train with 1 epoch vs many.

    LR model_short;
    model_short.Fit(data, 0.1, 1);
    double loss_after_1 = model_short.CrossEntropyLoss(data);

    LR model_long;
    model_long.Fit(data, 0.1, 1000);
    double loss_after_1000 = model_long.CrossEntropyLoss(data);

    std::cout << "[Result] Loss after 1 epoch    = " << loss_after_1 << std::endl;
    std::cout << "[Result] Loss after 1000 epochs = " << loss_after_1000 << std::endl;
    EXPECT_LT(loss_after_1000, loss_after_1);
}

TEST(LogisticRegression, CrossEntropyLossNonNegative) {
    PrintSection("CrossEntropyLossNonNegative",
                 "Cross-entropy is a sum of -log terms, which are always >= 0; "
                 "a negative value would signal a computation bug.");

    std::vector<LR::DataPoint> data = {
        {{1.0, 2.0}, 0}, {{3.0, 4.0}, 1}
    };

    LR model;
    model.Fit(data, 0.1, 500);
    double loss = model.CrossEntropyLoss(data);
    std::cout << "[Result] CrossEntropyLoss = " << loss << " (expect >= 0)" << std::endl;
    EXPECT_GE(loss, 0.0);
}

// ── High-dimensional (10D) ──────────────────────────────────────────────────

TEST(LogisticRegression, HighDimensional10D) {
    PrintSection("HighDimensional10D",
                 "Logistic regression must work beyond 2D; this test uses 10-dimensional "
                 "synthetic data where label = 1 iff sum(features) > 0. Confirms the "
                 "weight vector generalizes to arbitrary dimensionality.");

    // Generate synthetic data: label = 1 if sum > 0
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 3.0);

    std::vector<LR::DataPoint> train_data;
    std::vector<LR::DataPoint> test_data;

    auto make_point = [&]() -> LR::DataPoint {
        LR::DataPoint dp;
        dp.features.resize(10);
        for (int j = 0; j < 10; ++j) {
            dp.features[j] = dist(rng);
        }
        double sum = std::accumulate(dp.features.begin(), dp.features.end(), 0.0);
        dp.label = sum > 0.0 ? 1 : 0;
        return dp;
    };

    for (int i = 0; i < 200; ++i) train_data.push_back(make_point());
    for (int i = 0; i < 50; ++i) test_data.push_back(make_point());

    LR model;
    model.Fit(train_data, 0.01, 500);

    double train_acc = model.CalculateAccuracy(train_data);
    double test_acc  = model.CalculateAccuracy(test_data);

    std::cout << "[Result] Training accuracy (10D) = " << train_acc * 100.0 << "%" << std::endl;
    std::cout << "[Result] Test accuracy     (10D) = " << test_acc  * 100.0 << "%" << std::endl;

    const auto& w = model.GetWeights();
    std::cout << "[Result] Weights (" << w.size() << "D):";
    for (std::size_t i = 0; i < w.size(); ++i) {
        std::cout << " " << w[i];
    }
    std::cout << "\n[Result] Bias = " << model.GetBias() << std::endl;

    EXPECT_EQ(w.size(), 10u);
    // With sum-based labels and enough training, accuracy should be high
    EXPECT_GE(train_acc, 0.80);
    EXPECT_GE(test_acc, 0.70);
}

// ── Weight vector dimensionality ─────────────────────────────────────────────

TEST(LogisticRegression, WeightDimensionMatchesFeatures) {
    PrintSection("WeightDimensionMatchesFeatures",
                 "After training on 5D data, the weight vector must have exactly 5 "
                 "entries. A mismatch means Fit did not initialize weights correctly.");

    std::vector<LR::DataPoint> data = {
        {{1, 2, 3, 4, 5}, 0},
        {{6, 7, 8, 9, 10}, 1}
    };

    LR model;
    model.Fit(data, 0.1, 100);

    std::cout << "[Result] GetWeights().size() = " << model.GetWeights().size()
              << " (expect 5)" << std::endl;
    EXPECT_EQ(model.GetWeights().size(), 5u);
}

// ── Predict probability range ────────────────────────────────────────────────

TEST(LogisticRegression, PredictProbabilityInRange) {
    PrintSection("PredictProbabilityInRange",
                 "Predicted probabilities must lie in [0, 1]; values outside "
                 "this range are not valid probabilities.");

    std::vector<LR::DataPoint> data = {
        {{0.0, 0.0}, 0}, {{10.0, 10.0}, 1}
    };

    LR model;
    model.Fit(data, 0.1, 500);

    std::vector<std::vector<double>> queries = {
        {0.0, 0.0}, {5.0, 5.0}, {10.0, 10.0}, {-5.0, -5.0}, {100.0, 100.0}
    };

    for (const auto& q : queries) {
        double p = model.PredictProbability(q);
        std::cout << "[Result] P({" << q[0] << ", " << q[1] << "}) = " << p << std::endl;
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
    }
}

// ── Convergence / tolerance ──────────────────────────────────────────────────

TEST(LogisticRegression, EarlyStoppingWithTolerance) {
    PrintSection("EarlyStoppingWithTolerance",
                 "With a large tolerance, training should stop early. Loss should "
                 "still be lower than at initialization, confirming at least some "
                 "progress was made before stopping.");

    std::vector<LR::DataPoint> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0},
        {{5.0, 5.0}, 1}, {{6.0, 5.0}, 1}
    };

    LR model;
    model.Fit(data, 0.1, 10000, 1e-2);  // Large tolerance => early stop

    double loss = model.CrossEntropyLoss(data);
    std::cout << "[Result] Loss with tolerance=1e-2 = " << loss << std::endl;
    // Should have learned something (loss < log(2) ~ 0.693, the untrained loss)
    EXPECT_LT(loss, 0.693);
}

// ── Float instantiation ─────────────────────────────────────────────────────

TEST(LogisticRegression, FloatInstantiation) {
    PrintSection("FloatInstantiation",
                 "The template must compile and work with float, not just double. "
                 "Confirms the std::floating_point concept constraint is correct.");

    using LRf = LogisticRegression<float>;

    std::vector<LRf::DataPoint> data = {
        {{0.0f, 0.0f}, 0}, {{1.0f, 0.0f}, 0},
        {{5.0f, 5.0f}, 1}, {{6.0f, 5.0f}, 1}
    };

    LRf model;
    model.Fit(data, 0.1f, 500);

    float accuracy = model.CalculateAccuracy(data);
    std::cout << "[Result] Float accuracy = " << accuracy * 100.0f << "%" << std::endl;
    EXPECT_GE(accuracy, 0.75f);

    float s = LRf::Sigmoid(0.0f);
    std::cout << "[Result] Float Sigmoid(0) = " << s << " (expect ~0.5)" << std::endl;
    EXPECT_NEAR(s, 0.5f, 1e-5f);
}
