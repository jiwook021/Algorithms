/**
 * @file test_PrincipalComponentAnalysis.cpp
 * @brief Google Test suite for N-dimensional PrincipalComponentAnalysis.
 *
 * @details
 * Verifies:
 *   - Variance concentration along the principal axis
 *   - Dimensionality reduction (N -> K)
 *   - Explained variance ratios sum to 1
 *   - Orthogonality of principal components
 *   - Mean centering correctness
 *   - FitTransform equivalence
 *   - Float instantiation
 */

#include "PrincipalComponentAnalysis.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

// -----------------------------------------------------------------------
// Educational header printed before each logical test section.
// -----------------------------------------------------------------------
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

// -----------------------------------------------------------------------
// Type alias for convenience
// -----------------------------------------------------------------------
using PCA = PrincipalComponentAnalysis<double>;
using Matrix = PCA::Matrix;
using Vector = PCA::Vector;

// -----------------------------------------------------------------------
// Helper: compute dot product of two vectors
// -----------------------------------------------------------------------
double DotProduct(const Vector& a, const Vector& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// -----------------------------------------------------------------------
// Helper: compute vector norm
// -----------------------------------------------------------------------
double Norm(const Vector& v) {
    return std::sqrt(DotProduct(v, v));
}

// -----------------------------------------------------------------------
// Helper: compute variance of a 1D vector
// -----------------------------------------------------------------------
double Variance(const std::vector<double>& v) {
    double mean = 0.0;
    for (auto x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (auto x : v) var += (x - mean) * (x - mean);
    return var / static_cast<double>(v.size() - 1);
}

// -----------------------------------------------------------------------
// Test: Variance along principal axis
// -----------------------------------------------------------------------
TEST(PcaTest, VarianceAlongPrincipalAxis) {
    PrintSection("Variance along principal axis",
                 "The first principal component should capture the direction "
                 "of maximum variance. Projecting onto PC1 should yield "
                 "higher variance than projecting onto any other component.");

    // Data with strong linear trend: y = 2*x + noise
    Matrix data = {
        {1.0, 2.1}, {2.0, 3.9}, {3.0, 6.2}, {4.0, 7.8},
        {5.0, 10.1}, {6.0, 12.0}, {7.0, 13.9}, {8.0, 16.1}
    };

    PCA pca;
    pca.Fit(data);
    Matrix projected = pca.Transform(data, 2);

    // Extract PC1 and PC2 projections
    std::vector<double> pc1_vals, pc2_vals;
    for (const auto& row : projected) {
        pc1_vals.push_back(row[0]);
        pc2_vals.push_back(row[1]);
    }

    double var_pc1 = Variance(pc1_vals);
    double var_pc2 = Variance(pc2_vals);

    std::cout << "  Variance along PC1: " << var_pc1 << "\n";
    std::cout << "  Variance along PC2: " << var_pc2 << "\n";

    EXPECT_GT(var_pc1, var_pc2 * 10.0)
        << "PC1 should capture much more variance than PC2";
}

// -----------------------------------------------------------------------
// Test: Dimensionality reduction N -> K
// -----------------------------------------------------------------------
TEST(PcaTest, DimensionalityReduction) {
    PrintSection("Dimensionality reduction (N -> K)",
                 "PCA should reduce a 4D dataset to K dimensions while "
                 "preserving the number of samples. The output matrix "
                 "shape should be N_samples x K.");

    // 4D data: features 0 and 1 have variance, features 2 and 3 are near-constant
    Matrix data = {
        {1.0, 2.0, 5.0, 5.0},
        {2.0, 4.0, 5.1, 5.0},
        {3.0, 6.0, 4.9, 5.0},
        {4.0, 8.0, 5.0, 5.1},
        {5.0, 10.0, 5.1, 4.9},
    };

    PCA pca;
    Matrix projected = pca.FitTransform(data, 2);

    EXPECT_EQ(projected.size(), data.size())
        << "Number of samples should be preserved";
    EXPECT_EQ(projected[0].size(), 2u)
        << "Should reduce to 2 components";

    std::cout << "  Input shape:  " << data.size() << " x " << data[0].size() << "\n";
    std::cout << "  Output shape: " << projected.size() << " x "
              << projected[0].size() << "\n";
}

// -----------------------------------------------------------------------
// Test: Explained variance sums to 1
// -----------------------------------------------------------------------
TEST(PcaTest, ExplainedVarianceSumsToOne) {
    PrintSection("Explained variance sums to 1",
                 "The explained variance ratios represent the fraction of "
                 "total variance captured by each component. They must "
                 "sum to 1.0 (conservation of variance).");

    Matrix data = {
        {1.0, 2.0, 3.0},
        {2.0, 4.1, 2.9},
        {3.0, 5.9, 3.1},
        {4.0, 8.0, 2.8},
        {5.0, 10.1, 3.2},
        {6.0, 11.9, 2.7},
    };

    PCA pca;
    pca.Fit(data);
    const auto& ratios = pca.GetExplainedVarianceRatio();

    double total = 0.0;
    for (std::size_t i = 0; i < ratios.size(); ++i) {
        std::cout << "  PC" << (i + 1) << " explained variance ratio: "
                  << ratios[i] << "\n";
        total += ratios[i];
        EXPECT_GE(ratios[i], 0.0) << "Each ratio must be non-negative";
    }

    std::cout << "  Total: " << total << "\n";
    EXPECT_NEAR(total, 1.0, 1e-10)
        << "Explained variance ratios must sum to 1.0";
}

// -----------------------------------------------------------------------
// Test: Explained variance is in descending order
// -----------------------------------------------------------------------
TEST(PcaTest, ExplainedVarianceDescending) {
    PrintSection("Explained variance descending order",
                 "Components should be sorted by decreasing eigenvalue, "
                 "so explained variance ratios must be non-increasing.");

    Matrix data = {
        {1.0, 2.0, 3.0},
        {2.0, 3.5, 3.1},
        {3.0, 5.5, 2.9},
        {4.0, 7.0, 3.2},
        {5.0, 9.0, 2.8},
    };

    PCA pca;
    pca.Fit(data);
    const auto& ratios = pca.GetExplainedVarianceRatio();

    for (std::size_t i = 1; i < ratios.size(); ++i) {
        EXPECT_GE(ratios[i - 1], ratios[i])
            << "Ratios should be non-increasing";
    }
}

// -----------------------------------------------------------------------
// Test: Orthogonal components
// -----------------------------------------------------------------------
TEST(PcaTest, OrthogonalComponents) {
    PrintSection("Orthogonal components",
                 "Principal components are eigenvectors of a symmetric matrix "
                 "and must be mutually orthogonal (dot product ~ 0) and "
                 "unit-length (norm = 1).");

    Matrix data = {
        {1.0, 2.0, 3.0},
        {2.5, 3.5, 1.5},
        {3.0, 6.0, 2.0},
        {4.5, 7.5, 3.5},
        {5.0, 10.0, 1.0},
        {6.5, 11.5, 2.5},
    };

    PCA pca;
    pca.Fit(data);
    const auto& components = pca.GetComponents();

    std::size_t n = components.size();
    for (std::size_t i = 0; i < n; ++i) {
        // Check unit length
        double norm = Norm(components[i]);
        std::cout << "  |PC" << (i + 1) << "| = " << norm << "\n";
        EXPECT_NEAR(norm, 1.0, 1e-8)
            << "Component " << i << " should be unit-length";

        // Check orthogonality with all other components
        for (std::size_t j = i + 1; j < n; ++j) {
            double dot = DotProduct(components[i], components[j]);
            std::cout << "  PC" << (i + 1) << " . PC" << (j + 1)
                      << " = " << dot << "\n";
            EXPECT_NEAR(dot, 0.0, 1e-8)
                << "Components " << i << " and " << j
                << " should be orthogonal";
        }
    }
}

// -----------------------------------------------------------------------
// Test: Mean centering
// -----------------------------------------------------------------------
TEST(PcaTest, MeanCentering) {
    PrintSection("Mean centering",
                 "PCA subtracts the per-feature mean before computing "
                 "covariance. The stored mean should match the data mean.");

    Matrix data = {
        {2.0, 4.0},
        {4.0, 8.0},
        {6.0, 12.0},
    };

    PCA pca;
    pca.Fit(data);
    const auto& mean = pca.GetMean();

    EXPECT_NEAR(mean[0], 4.0, 1e-10);
    EXPECT_NEAR(mean[1], 8.0, 1e-10);

    std::cout << "  Computed mean: (" << mean[0] << ", " << mean[1] << ")\n";
    std::cout << "  Expected mean: (4.0, 8.0)\n";
}

// -----------------------------------------------------------------------
// Test: Projection of mean yields zero
// -----------------------------------------------------------------------
TEST(PcaTest, MeanProjectsToZero) {
    PrintSection("Mean projects to zero",
                 "After centering, projecting the data mean onto any "
                 "principal component should yield zero.");

    Matrix data = {
        {1.0, 2.0, 3.0},
        {3.0, 4.0, 5.0},
        {5.0, 6.0, 7.0},
    };

    PCA pca;
    pca.Fit(data);
    const auto& mean = pca.GetMean();

    // Transform the mean point
    Matrix mean_as_data = {mean};
    Matrix projected = pca.Transform(mean_as_data, 3);

    for (std::size_t k = 0; k < projected[0].size(); ++k) {
        std::cout << "  Mean projected onto PC" << (k + 1)
                  << ": " << projected[0][k] << "\n";
        EXPECT_NEAR(projected[0][k], 0.0, 1e-10)
            << "Mean projection onto PC" << (k + 1) << " should be zero";
    }
}

// -----------------------------------------------------------------------
// Test: FitTransform equivalence
// -----------------------------------------------------------------------
TEST(PcaTest, FitTransformEquivalence) {
    PrintSection("FitTransform equivalence",
                 "FitTransform(data, k) must produce the same result as "
                 "calling Fit(data) then Transform(data, k) separately.");

    Matrix data = {
        {1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {4.0, 8.0},
    };

    PCA pca1;
    pca1.Fit(data);
    Matrix result1 = pca1.Transform(data, 1);

    PCA pca2;
    Matrix result2 = pca2.FitTransform(data, 1);

    ASSERT_EQ(result1.size(), result2.size());
    for (std::size_t i = 0; i < result1.size(); ++i) {
        ASSERT_EQ(result1[i].size(), result2[i].size());
        for (std::size_t j = 0; j < result1[i].size(); ++j) {
            EXPECT_NEAR(result1[i][j], result2[i][j], 1e-10)
                << "Results should match at (" << i << ", " << j << ")";
        }
    }

    std::cout << "  Fit+Transform and FitTransform produce identical results.\n";
}

// -----------------------------------------------------------------------
// Test: 2D diagonal data direction (regression for original 2D behavior)
// -----------------------------------------------------------------------
TEST(PcaTest, DiagonalDataDirection) {
    PrintSection("Diagonal data (y = x) direction",
                 "For perfectly correlated y = x data, the first principal "
                 "component should point along the (1/sqrt(2), 1/sqrt(2)) "
                 "direction (45 degrees).");

    Matrix data = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}};

    PCA pca;
    pca.Fit(data);
    const auto& components = pca.GetComponents();

    double abs_x = std::abs(components[0][0]);
    double abs_y = std::abs(components[0][1]);

    std::cout << "  PC1 direction: (" << components[0][0] << ", "
              << components[0][1] << ")\n";
    std::cout << "  Expected |x| ~ |y| ~ 0.7071\n";

    EXPECT_NEAR(abs_x, abs_y, 0.01)
        << "Diagonal data should give 45-degree principal component";
}

// -----------------------------------------------------------------------
// Test: Float instantiation
// -----------------------------------------------------------------------
TEST(PcaTest, FloatInstantiation) {
    PrintSection("Float instantiation",
                 "The template should compile and work correctly with float "
                 "precision, not just double.");

    using PCAf = PrincipalComponentAnalysis<float>;
    PCAf::Matrix data = {
        {1.0f, 2.0f}, {2.0f, 4.0f}, {3.0f, 6.0f},
    };

    PCAf pca;
    pca.Fit(data);
    PCAf::Matrix projected = pca.Transform(data, 1);

    EXPECT_EQ(projected.size(), 3u);
    EXPECT_EQ(projected[0].size(), 1u);

    const auto& ratios = pca.GetExplainedVarianceRatio();
    float total = 0.0f;
    for (auto r : ratios) total += r;
    EXPECT_NEAR(total, 1.0f, 1e-5f);

    std::cout << "  Float PCA: explained variance sum = " << total << "\n";
}

// -----------------------------------------------------------------------
// Test: Error handling - unfitted transform
// -----------------------------------------------------------------------
TEST(PcaTest, TransformBeforeFitThrows) {
    PrintSection("Error handling: transform before fit",
                 "Calling Transform without first calling Fit should throw "
                 "a runtime_error to prevent undefined behavior.");

    PCA pca;
    Matrix data = {{1.0, 2.0}};

    EXPECT_THROW(pca.Transform(data, 1), std::runtime_error);
    std::cout << "  Correctly throws runtime_error for unfitted model.\n";
}

// -----------------------------------------------------------------------
// Test: Error handling - empty data
// -----------------------------------------------------------------------
TEST(PcaTest, EmptyDataThrows) {
    PrintSection("Error handling: empty data",
                 "Fitting on empty data should throw an invalid_argument "
                 "since PCA requires at least one sample.");

    PCA pca;
    Matrix data = {};

    EXPECT_THROW(pca.Fit(data), std::invalid_argument);
    std::cout << "  Correctly throws invalid_argument for empty data.\n";
}

}  // namespace
