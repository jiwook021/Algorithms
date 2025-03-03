/**
 * @file test_KMeansClustering.cpp
 * @brief Google Test suite for N-dimensional KMeansClustering.
 *
 * Each test prints an educational banner explaining *what* is being tested
 * and *why*, followed by [Result] lines showing computed vs expected values
 * before the assertion.
 *
 * Coverage:
 *   - Euclidean distance in N dimensions
 *   - Convergence on well-separated 2D clusters
 *   - Inertia decreases with more clusters
 *   - High-dimensional clustering (5D)
 *   - Single-cluster degenerate case
 *   - Predict routes new points to the correct cluster
 *   - Float instantiation compiles and converges
 */

#include "KMeansClustering.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: educational banner printed at the start of every test
// ---------------------------------------------------------------------------

namespace {

void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

} // namespace

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

using KMD    = KMeansClustering<double>;
using Vector = KMD::Vector;
using Matrix = KMD::Matrix;

static constexpr double kTol = 1e-6;

// ===========================================================================
// Distance tests
// ===========================================================================

TEST(KMeansTest, Distance2D) {
    PrintSection("Euclidean Distance 2D",
                 "Validates the fundamental distance metric on a classic "
                 "3-4-5 right triangle.");

    double d = KMD::Distance({0.0, 0.0}, {3.0, 4.0});
    std::cout << "[Result] Distance((0,0),(3,4)) = " << d
              << "  (expected 5.0)" << std::endl;
    EXPECT_NEAR(d, 5.0, kTol);
}

TEST(KMeansTest, Distance3D) {
    PrintSection("Euclidean Distance 3D",
                 "Ensures the distance generalizes beyond 2D correctly.");

    double d = KMD::Distance({1.0, 2.0, 3.0}, {4.0, 6.0, 3.0});
    double expected = std::sqrt(9.0 + 16.0); // 5.0
    std::cout << "[Result] Distance((1,2,3),(4,6,3)) = " << d
              << "  (expected " << expected << ")" << std::endl;
    EXPECT_NEAR(d, expected, kTol);
}

// ===========================================================================
// Well-separated 2D clusters -- must converge
// ===========================================================================

TEST(KMeansTest, ConvergesOnWellSeparatedClusters) {
    PrintSection("Convergence on Well-Separated Clusters",
                 "Two tight groups far apart should produce two clusters "
                 "with every point in the correct group.");

    Matrix data = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
        {100.0, 100.0}, {101.0, 100.0}, {100.0, 101.0}
    };

    KMD model(2, 100, 1e-6, 42);
    auto assignments = model.Fit(data);

    // First three points must share a cluster; last three share another.
    std::cout << "[Result] assignments: ";
    for (int a : assignments) std::cout << a << " ";
    std::cout << std::endl;

    EXPECT_EQ(assignments[0], assignments[1]);
    EXPECT_EQ(assignments[1], assignments[2]);
    EXPECT_EQ(assignments[3], assignments[4]);
    EXPECT_EQ(assignments[4], assignments[5]);
    EXPECT_NE(assignments[0], assignments[3]);
}

// ===========================================================================
// Inertia decreases with more clusters
// ===========================================================================

TEST(KMeansTest, InertiaDecreasesWithMoreClusters) {
    PrintSection("Inertia Decreases",
                 "Adding more clusters should never increase inertia (WCSS). "
                 "This is a fundamental monotonicity property of K-Means.");

    Matrix data = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
        {10.0, 10.0}, {11.0, 10.0}, {10.0, 11.0},
        {20.0, 0.0}, {21.0, 0.0}, {20.0, 1.0}
    };

    KMD model2(2, 200, 1e-8, 42);
    model2.Fit(data);
    double inertia2 = model2.Inertia();

    KMD model3(3, 200, 1e-8, 42);
    model3.Fit(data);
    double inertia3 = model3.Inertia();

    std::cout << "[Result] Inertia(k=2) = " << inertia2 << std::endl;
    std::cout << "[Result] Inertia(k=3) = " << inertia3 << std::endl;

    EXPECT_LT(inertia3, inertia2 + kTol);
}

// ===========================================================================
// High-dimensional clustering (5D)
// ===========================================================================

TEST(KMeansTest, HighDimensional5D) {
    PrintSection("High-Dimensional Clustering (5D)",
                 "K-Means must work correctly in arbitrary dimensions, "
                 "not just 2D. We create three well-separated 5D clusters.");

    // Three clusters centred at (0,...,0), (50,...,50), (100,...,100)
    Matrix data;
    for (int i = 0; i < 10; ++i) {
        data.push_back({0.0 + i * 0.1, 0.0 + i * 0.1, 0.0 + i * 0.1,
                        0.0 + i * 0.1, 0.0 + i * 0.1});
    }
    for (int i = 0; i < 10; ++i) {
        data.push_back({50.0 + i * 0.1, 50.0 + i * 0.1, 50.0 + i * 0.1,
                        50.0 + i * 0.1, 50.0 + i * 0.1});
    }
    for (int i = 0; i < 10; ++i) {
        data.push_back({100.0 + i * 0.1, 100.0 + i * 0.1, 100.0 + i * 0.1,
                        100.0 + i * 0.1, 100.0 + i * 0.1});
    }

    KMD model(3, 200, 1e-8, 42);
    auto assignments = model.Fit(data);

    // Points 0..9 must share a cluster, 10..19, 20..29 likewise
    int c0 = assignments[0];
    int c1 = assignments[10];
    int c2 = assignments[20];

    std::cout << "[Result] Cluster labels for groups: "
              << c0 << ", " << c1 << ", " << c2 << std::endl;

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(assignments[i], c0) << "Point " << i;
    }
    for (int i = 10; i < 20; ++i) {
        EXPECT_EQ(assignments[i], c1) << "Point " << i;
    }
    for (int i = 20; i < 30; ++i) {
        EXPECT_EQ(assignments[i], c2) << "Point " << i;
    }

    // All three cluster labels must be distinct
    EXPECT_NE(c0, c1);
    EXPECT_NE(c1, c2);
    EXPECT_NE(c0, c2);
}

// ===========================================================================
// Single cluster (k=1)
// ===========================================================================

TEST(KMeansTest, SingleCluster) {
    PrintSection("Single Cluster (k=1)",
                 "When k=1, all points must be assigned to cluster 0 and "
                 "the centroid is the data mean.");

    Matrix data = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
    KMD model(1, 100, 1e-6, 42);
    auto assignments = model.Fit(data);

    for (std::size_t i = 0; i < assignments.size(); ++i) {
        EXPECT_EQ(assignments[i], 0) << "Point " << i;
    }

    const auto& centroids = model.GetCentroids();
    ASSERT_EQ(centroids.size(), 1u);
    std::cout << "[Result] Centroid = (" << centroids[0][0]
              << ", " << centroids[0][1] << ")  (expected 2, 2)" << std::endl;
    EXPECT_NEAR(centroids[0][0], 2.0, kTol);
    EXPECT_NEAR(centroids[0][1], 2.0, kTol);
}

// ===========================================================================
// Predict assigns to nearest centroid
// ===========================================================================

TEST(KMeansTest, PredictAssignsCorrectly) {
    PrintSection("Predict",
                 "After fitting, Predict must assign a new point to the "
                 "nearest centroid without re-running the full algorithm.");

    Matrix data = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0},
        {10.0, 10.0}, {11.0, 10.0}, {10.0, 11.0}
    };

    KMD model(2, 100, 1e-6, 42);
    auto assignments = model.Fit(data);

    int cluster_near_origin = assignments[0];
    int cluster_far = assignments[3];

    int pred_near = model.Predict({0.5, 0.5});
    int pred_far  = model.Predict({10.5, 10.5});

    std::cout << "[Result] Predict(0.5,0.5) = " << pred_near
              << "  (expected " << cluster_near_origin << ")" << std::endl;
    std::cout << "[Result] Predict(10.5,10.5) = " << pred_far
              << "  (expected " << cluster_far << ")" << std::endl;

    EXPECT_EQ(pred_near, cluster_near_origin);
    EXPECT_EQ(pred_far, cluster_far);
}

// ===========================================================================
// Float instantiation
// ===========================================================================

TEST(KMeansTest, FloatInstantiation) {
    PrintSection("Float Instantiation",
                 "The template must compile and converge with float, "
                 "not just double.");

    using KMF = KMeansClustering<float>;
    KMF::Matrix data = {
        {0.0f, 0.0f}, {1.0f, 0.0f},
        {100.0f, 100.0f}, {101.0f, 100.0f}
    };

    KMF model(2, 100, 1e-4f, 42);
    auto assignments = model.Fit(data);

    EXPECT_EQ(assignments[0], assignments[1]);
    EXPECT_EQ(assignments[2], assignments[3]);
    EXPECT_NE(assignments[0], assignments[2]);

    float inertia = model.Inertia();
    std::cout << "[Result] Float inertia = " << inertia << std::endl;
    EXPECT_GT(inertia, 0.0f);
}

// ===========================================================================
// Inertia is non-negative
// ===========================================================================

TEST(KMeansTest, InertiaIsNonNegative) {
    PrintSection("Inertia Non-Negative",
                 "Inertia (sum of squared distances) is always >= 0.");

    Matrix data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    KMD model(2, 100, 1e-6, 42);
    model.Fit(data);
    double inertia = model.Inertia();

    std::cout << "[Result] Inertia = " << inertia << std::endl;
    EXPECT_GE(inertia, 0.0);
}
