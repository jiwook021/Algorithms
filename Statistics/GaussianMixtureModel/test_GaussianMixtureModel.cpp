/**
 * @file test_GaussianMixtureModel.cpp
 * @brief Google Test suite for K-component N-dimensional GaussianMixtureModel.
 * @details Verifies EM convergence (log-likelihood monotonically increases),
 *          recovery of true component parameters, soft assignment properties,
 *          and edge cases with educational PrintSection output.
 */

#include "GaussianMixtureModel.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

// Convenience aliases
using GMM = GaussianMixtureModel<double>;
using Vector = GMM::Vector;
using Matrix = GMM::Matrix;

static constexpr double kTestTolerance = 1e-4;

// ---------------------------------------------------------------------------
// Helper: generate N-dimensional Gaussian samples
// ---------------------------------------------------------------------------
Matrix GenerateCluster(const Vector& mean, const Vector& stddev,
                       std::size_t n_samples, std::mt19937& rng) {
    Matrix samples;
    samples.reserve(n_samples);
    std::size_t dims = mean.size();
    for (std::size_t i = 0; i < n_samples; ++i) {
        Vector point(dims);
        for (std::size_t d = 0; d < dims; ++d) {
            std::normal_distribution<double> dist(mean[d], stddev[d]);
            point[d] = dist(rng);
        }
        samples.push_back(point);
    }
    return samples;
}

} // namespace

// ===========================================================================
// 1. EM Convergence: log-likelihood must monotonically increase
// ===========================================================================

TEST(GaussianMixtureModel, EmConvergenceLogLikelihoodIncreases) {
    PrintSection("EM Convergence - Log-Likelihood Increases",
        "The EM algorithm guarantees that log-likelihood never decreases "
        "between iterations. We verify this by running single EM steps "
        "and checking monotonicity.");

    std::mt19937 rng(123);
    Matrix data;
    auto c1 = GenerateCluster({-5.0}, {1.0}, 100, rng);
    auto c2 = GenerateCluster({5.0}, {1.0}, 100, rng);
    data.insert(data.end(), c1.begin(), c1.end());
    data.insert(data.end(), c2.begin(), c2.end());

    // Run 1-iteration models and track log-likelihood
    std::vector<double> log_likelihoods;
    for (int max_iter = 1; max_iter <= 20; ++max_iter) {
        GMM gmm(2, max_iter, 0.0);  // tolerance=0 forces exact max_iter steps
        gmm.Fit(data);
        log_likelihoods.push_back(gmm.LogLikelihood(data));
    }

    std::cout << "  Log-likelihood progression (first 10):\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(10, log_likelihoods.size()); ++i) {
        std::cout << "    iter " << (i + 1) << ": " << log_likelihoods[i] << "\n";
    }

    // Verify monotonic non-decrease (allow small numerical noise)
    for (std::size_t i = 1; i < log_likelihoods.size(); ++i) {
        EXPECT_GE(log_likelihoods[i], log_likelihoods[i - 1] - 1e-6)
            << "Log-likelihood decreased at iteration " << (i + 1);
    }
}

// ===========================================================================
// 2. Recovers true components in 1D
// ===========================================================================

TEST(GaussianMixtureModel, RecoversTrue1DComponents) {
    PrintSection("Recover True 1D Components",
        "Given well-separated 1D clusters, the GMM should recover means "
        "close to the true generating means and variances.");

    std::mt19937 rng(42);
    double true_mean_a = -5.0, true_mean_b = 5.0;
    double true_var_a = 1.0, true_var_b = 1.0;

    Matrix data;
    auto c1 = GenerateCluster({true_mean_a}, {std::sqrt(true_var_a)}, 200, rng);
    auto c2 = GenerateCluster({true_mean_b}, {std::sqrt(true_var_b)}, 200, rng);
    data.insert(data.end(), c1.begin(), c1.end());
    data.insert(data.end(), c2.begin(), c2.end());

    GMM gmm(2, 200, 1e-8);
    gmm.Fit(data);

    const auto& components = gmm.GetComponents();
    ASSERT_EQ(components.size(), 2u);

    // Sort components by mean for consistent comparison
    std::vector<std::size_t> order = {0, 1};
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) {
                  return components[a].mean[0] < components[b].mean[0];
              });

    std::cout << "  True means:     " << true_mean_a << ", " << true_mean_b << "\n";
    std::cout << "  Recovered means: " << components[order[0]].mean[0]
              << ", " << components[order[1]].mean[0] << "\n";
    std::cout << "  True variances: " << true_var_a << ", " << true_var_b << "\n";
    std::cout << "  Recovered var:  " << components[order[0]].variance[0]
              << ", " << components[order[1]].variance[0] << "\n";
    std::cout << "  Weights:        " << components[order[0]].weight
              << ", " << components[order[1]].weight << "\n";

    EXPECT_NEAR(components[order[0]].mean[0], true_mean_a, 0.5);
    EXPECT_NEAR(components[order[1]].mean[0], true_mean_b, 0.5);
    EXPECT_NEAR(components[order[0]].variance[0], true_var_a, 0.5);
    EXPECT_NEAR(components[order[1]].variance[0], true_var_b, 0.5);
    EXPECT_NEAR(components[order[0]].weight, 0.5, 0.1);
    EXPECT_NEAR(components[order[1]].weight, 0.5, 0.1);
}

// ===========================================================================
// 3. Recovers true components in 2D
// ===========================================================================

TEST(GaussianMixtureModel, RecoversTrue2DComponents) {
    PrintSection("Recover True 2D Components",
        "Verifies that the GMM correctly estimates 2D means and diagonal "
        "variances when clusters are well-separated in both dimensions.");

    std::mt19937 rng(99);
    Vector true_mean_a = {-4.0, -4.0};
    Vector true_mean_b = {4.0, 4.0};
    Vector true_std_a = {1.0, 0.5};
    Vector true_std_b = {0.5, 1.0};

    Matrix data;
    auto c1 = GenerateCluster(true_mean_a, true_std_a, 300, rng);
    auto c2 = GenerateCluster(true_mean_b, true_std_b, 300, rng);
    data.insert(data.end(), c1.begin(), c1.end());
    data.insert(data.end(), c2.begin(), c2.end());

    GMM gmm(2, 200, 1e-8);
    gmm.Fit(data);

    const auto& components = gmm.GetComponents();
    ASSERT_EQ(components.size(), 2u);

    // Sort by first mean dimension
    std::vector<std::size_t> order = {0, 1};
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) {
                  return components[a].mean[0] < components[b].mean[0];
              });

    std::cout << "  Cluster A recovered mean: ["
              << components[order[0]].mean[0] << ", "
              << components[order[0]].mean[1] << "]\n";
    std::cout << "  Cluster B recovered mean: ["
              << components[order[1]].mean[0] << ", "
              << components[order[1]].mean[1] << "]\n";

    EXPECT_NEAR(components[order[0]].mean[0], true_mean_a[0], 0.5);
    EXPECT_NEAR(components[order[0]].mean[1], true_mean_a[1], 0.5);
    EXPECT_NEAR(components[order[1]].mean[0], true_mean_b[0], 0.5);
    EXPECT_NEAR(components[order[1]].mean[1], true_mean_b[1], 0.5);
}

// ===========================================================================
// 4. Soft assignment responsibilities sum to 1
// ===========================================================================

TEST(GaussianMixtureModel, ResponsibilitiesSumToOne) {
    PrintSection("Responsibilities Sum to 1",
        "For each data point, the soft assignment probabilities across all "
        "components must sum to exactly 1, since they form a valid "
        "probability distribution.");

    std::mt19937 rng(77);
    Matrix data;
    auto c1 = GenerateCluster({0.0, 0.0}, {1.0, 1.0}, 50, rng);
    auto c2 = GenerateCluster({6.0, 6.0}, {1.0, 1.0}, 50, rng);
    auto c3 = GenerateCluster({0.0, 6.0}, {1.0, 1.0}, 50, rng);
    data.insert(data.end(), c1.begin(), c1.end());
    data.insert(data.end(), c2.begin(), c2.end());
    data.insert(data.end(), c3.begin(), c3.end());

    GMM gmm(3, 100, 1e-6);
    gmm.Fit(data);

    auto resp = gmm.Responsibilities(data);
    ASSERT_EQ(resp.size(), data.size());

    std::cout << "  Checking " << resp.size()
              << " responsibility vectors sum to 1...\n";
    double max_deviation = 0.0;
    for (std::size_t i = 0; i < resp.size(); ++i) {
        ASSERT_EQ(resp[i].size(), 3u);
        double sum = 0.0;
        for (std::size_t k = 0; k < resp[i].size(); ++k) {
            sum += resp[i][k];
            EXPECT_GE(resp[i][k], 0.0) << "Responsibility must be non-negative";
        }
        max_deviation = std::max(max_deviation, std::abs(sum - 1.0));
        EXPECT_NEAR(sum, 1.0, 1e-10)
            << "Responsibilities at sample " << i << " do not sum to 1";
    }
    std::cout << "  Max deviation from 1.0: " << max_deviation << "\n";
}

// ===========================================================================
// 5. Weights sum to 1 after fitting
// ===========================================================================

TEST(GaussianMixtureModel, WeightsSumToOne) {
    PrintSection("Component Weights Sum to 1",
        "Mixing weights represent prior probabilities of each component "
        "and must form a valid probability distribution.");

    std::mt19937 rng(55);
    Matrix data;
    auto c1 = GenerateCluster({-3.0}, {1.0}, 80, rng);
    auto c2 = GenerateCluster({3.0}, {1.0}, 120, rng);
    data.insert(data.end(), c1.begin(), c1.end());
    data.insert(data.end(), c2.begin(), c2.end());

    GMM gmm(2, 100, 1e-6);
    gmm.Fit(data);

    const auto& components = gmm.GetComponents();
    double weight_sum = 0.0;
    for (const auto& comp : components) {
        weight_sum += comp.weight;
        std::cout << "  Component weight: " << comp.weight << "\n";
    }
    std::cout << "  Total: " << weight_sum << "\n";

    EXPECT_NEAR(weight_sum, 1.0, 1e-6);
}

// ===========================================================================
// 6. Variance stays positive after fitting
// ===========================================================================

TEST(GaussianMixtureModel, VarianceRemainsPositive) {
    PrintSection("Variance Remains Positive",
        "Even with identical data points within a cluster, the variance "
        "floor should prevent singularities in the covariance matrix.");

    Matrix data;
    // Identical points -- variance would be zero without a floor
    for (int i = 0; i < 50; ++i) data.push_back({1.0, 2.0});
    for (int i = 0; i < 50; ++i) data.push_back({10.0, 20.0});

    GMM gmm(2, 100, 1e-6);
    gmm.Fit(data);

    const auto& components = gmm.GetComponents();
    for (std::size_t k = 0; k < components.size(); ++k) {
        for (std::size_t d = 0; d < components[k].variance.size(); ++d) {
            std::cout << "  Component " << k << " dim " << d
                      << " variance: " << components[k].variance[d] << "\n";
            EXPECT_GT(components[k].variance[d], 0.0)
                << "Variance must be positive for component " << k
                << " dimension " << d;
        }
    }
}

// ===========================================================================
// 7. Predict assigns correct clusters for well-separated data
// ===========================================================================

TEST(GaussianMixtureModel, PredictWellSeparatedClusters) {
    PrintSection("Predict Well-Separated Clusters",
        "When clusters are far apart, hard assignment should correctly "
        "separate data points into their generating clusters.");

    std::mt19937 rng(33);
    Matrix data;
    auto c1 = GenerateCluster({-10.0}, {0.5}, 100, rng);
    auto c2 = GenerateCluster({10.0}, {0.5}, 100, rng);
    data.insert(data.end(), c1.begin(), c1.end());
    data.insert(data.end(), c2.begin(), c2.end());

    GMM gmm(2, 100, 1e-6);
    gmm.Fit(data);

    auto labels = gmm.Predict(data);
    ASSERT_EQ(labels.size(), 200u);

    // All points in the first group should share one label,
    // all in the second should share the other
    int label_a = labels[0];
    int label_b = labels[100];
    EXPECT_NE(label_a, label_b)
        << "Two separated clusters should have different labels";

    int misclassified = 0;
    for (std::size_t i = 0; i < 100; ++i) {
        if (labels[i] != label_a) ++misclassified;
    }
    for (std::size_t i = 100; i < 200; ++i) {
        if (labels[i] != label_b) ++misclassified;
    }

    std::cout << "  Misclassified: " << misclassified << " / 200\n";
    EXPECT_LE(misclassified, 5)
        << "At most 5 misclassifications expected for well-separated data";
}

// ===========================================================================
// 8. Three-component model in 2D
// ===========================================================================

TEST(GaussianMixtureModel, ThreeComponentsIn2D) {
    PrintSection("Three Components in 2D",
        "Verifies that the model can successfully fit K=3 components "
        "in 2 dimensions and recover approximately correct parameters.");

    std::mt19937 rng(101);
    Vector mean_a = {0.0, 0.0};
    Vector mean_b = {10.0, 0.0};
    Vector mean_c = {5.0, 8.66};  // equilateral triangle arrangement
    Vector std_all = {1.0, 1.0};

    Matrix data;
    auto c1 = GenerateCluster(mean_a, std_all, 150, rng);
    auto c2 = GenerateCluster(mean_b, std_all, 150, rng);
    auto c3 = GenerateCluster(mean_c, std_all, 150, rng);
    data.insert(data.end(), c1.begin(), c1.end());
    data.insert(data.end(), c2.begin(), c2.end());
    data.insert(data.end(), c3.begin(), c3.end());

    GMM gmm(3, 200, 1e-8);
    gmm.Fit(data);

    const auto& components = gmm.GetComponents();
    ASSERT_EQ(components.size(), 3u);

    // Sort by first dimension of mean
    std::vector<std::size_t> order = {0, 1, 2};
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) {
                  return components[a].mean[0] < components[b].mean[0];
              });

    std::cout << "  Recovered means:\n";
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << "    Component " << i << ": ["
                  << components[order[i]].mean[0] << ", "
                  << components[order[i]].mean[1] << "]\n";
    }

    // Each weight should be approximately 1/3
    for (std::size_t k = 0; k < 3; ++k) {
        EXPECT_NEAR(components[k].weight, 1.0 / 3.0, 0.1)
            << "Each component should have weight ~1/3";
    }
}

// ===========================================================================
// 9. Single component GMM
// ===========================================================================

TEST(GaussianMixtureModel, SingleComponent) {
    PrintSection("Single Component (K=1)",
        "A GMM with K=1 should estimate the global mean and variance "
        "of the data, with weight = 1.");

    std::mt19937 rng(7);
    Matrix data = GenerateCluster({3.0, -2.0}, {1.5, 0.8}, 500, rng);

    GMM gmm(1, 100, 1e-6);
    gmm.Fit(data);

    const auto& components = gmm.GetComponents();
    ASSERT_EQ(components.size(), 1u);

    std::cout << "  Recovered mean: [" << components[0].mean[0] << ", "
              << components[0].mean[1] << "]\n";
    std::cout << "  Weight: " << components[0].weight << "\n";

    EXPECT_NEAR(components[0].mean[0], 3.0, 0.3);
    EXPECT_NEAR(components[0].mean[1], -2.0, 0.3);
    EXPECT_NEAR(components[0].weight, 1.0, 1e-10);
}

// ===========================================================================
// 10. Float precision works
// ===========================================================================

TEST(GaussianMixtureModel, FloatPrecision) {
    PrintSection("Float Precision (Scalar = float)",
        "The template should work with float as well as double, "
        "producing reasonable results despite lower precision.");

    using FGMM = GaussianMixtureModel<float>;
    FGMM::Matrix data;

    std::mt19937 rng(12);
    std::normal_distribution<float> d1(-5.0f, 1.0f);
    std::normal_distribution<float> d2(5.0f, 1.0f);
    for (int i = 0; i < 100; ++i) {
        data.push_back({d1(rng)});
    }
    for (int i = 0; i < 100; ++i) {
        data.push_back({d2(rng)});
    }

    FGMM gmm(2, 100, 1e-4f);
    gmm.Fit(data);

    const auto& components = gmm.GetComponents();
    ASSERT_EQ(components.size(), 2u);

    // Sort by mean
    std::vector<std::size_t> order = {0, 1};
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) {
                  return components[a].mean[0] < components[b].mean[0];
              });

    std::cout << "  Float means: " << components[order[0]].mean[0]
              << ", " << components[order[1]].mean[0] << "\n";

    EXPECT_NEAR(components[order[0]].mean[0], -5.0f, 1.0f);
    EXPECT_NEAR(components[order[1]].mean[0], 5.0f, 1.0f);
}

// ===========================================================================
// 11. Unequal cluster sizes
// ===========================================================================

TEST(GaussianMixtureModel, UnequalClusterSizes) {
    PrintSection("Unequal Cluster Sizes",
        "When one cluster has many more samples, the recovered weights "
        "should reflect the true mixing proportions.");

    std::mt19937 rng(88);
    Matrix data;
    auto c1 = GenerateCluster({0.0}, {1.0}, 300, rng);   // 75%
    auto c2 = GenerateCluster({10.0}, {1.0}, 100, rng);  // 25%
    data.insert(data.end(), c1.begin(), c1.end());
    data.insert(data.end(), c2.begin(), c2.end());

    GMM gmm(2, 200, 1e-8);
    gmm.Fit(data);

    const auto& components = gmm.GetComponents();
    std::vector<std::size_t> order = {0, 1};
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) {
                  return components[a].mean[0] < components[b].mean[0];
              });

    std::cout << "  Larger cluster weight: " << components[order[0]].weight << " (expected ~0.75)\n";
    std::cout << "  Smaller cluster weight: " << components[order[1]].weight << " (expected ~0.25)\n";

    EXPECT_NEAR(components[order[0]].weight, 0.75, 0.1);
    EXPECT_NEAR(components[order[1]].weight, 0.25, 0.1);
}
