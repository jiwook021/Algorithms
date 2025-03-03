/**
 * @file test_KNearestNeighbors.cpp
 * @brief Google Test suite for the N-dimensional KNearestNeighbors classifier.
 *
 * Each test prints an educational banner explaining *why* it matters.
 */

#include "KNearestNeighbors.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

/**
 * @brief Print a section banner for educational output.
 * @param title Test section title.
 * @param why   One-line explanation of why this test matters.
 */
void PrintSection(const std::string& title, const std::string& why) {
    std::cout << "\n======= " << title << " =======\n"
              << "Why: " << why << "\n" << std::endl;
}

using KNN = KNearestNeighbors<double>;

} // namespace

// ---------------------------------------------------------------------------
// Majority vote
// ---------------------------------------------------------------------------

TEST(KNearestNeighborsTest, MajorityVoteTwoClusters) {
    PrintSection("Majority Vote -- Two Clusters",
                 "KNN classifies by majority vote of K nearest neighbors. "
                 "Points near a cluster should receive that cluster's label.");

    std::vector<KNN::DataPoint> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0}, {{0.0, 1.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}, {{10.0, 11.0}, 1}
    };

    KNN model(3);
    model.Fit(data);

    EXPECT_EQ(model.Predict({0.5, 0.5}), 0)
        << "Query near class-0 cluster must predict 0";
    EXPECT_EQ(model.Predict({10.5, 10.5}), 1)
        << "Query near class-1 cluster must predict 1";

    std::cout << "  Predict({0.5, 0.5})   = 0  (correct)\n";
    std::cout << "  Predict({10.5,10.5})  = 1  (correct)\n";
}

TEST(KNearestNeighborsTest, MajorityVoteMultiClass) {
    PrintSection("Majority Vote -- Multi-Class",
                 "KNN is not limited to binary labels. Majority vote must "
                 "work for three or more classes.");

    std::vector<KNN::DataPoint> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0}, {{0.0, 1.0}, 0},
        {{10.0, 0.0}, 1}, {{11.0, 0.0}, 1}, {{10.0, 1.0}, 1},
        {{0.0, 10.0}, 2}, {{1.0, 10.0}, 2}, {{0.0, 11.0}, 2}
    };

    KNN model(3);
    model.Fit(data);

    EXPECT_EQ(model.Predict({0.5, 0.5}), 0);
    EXPECT_EQ(model.Predict({10.5, 0.5}), 1);
    EXPECT_EQ(model.Predict({0.5, 10.5}), 2);

    std::cout << "  Three classes correctly identified by majority vote.\n";
}

// ---------------------------------------------------------------------------
// K = 1  nearest neighbor
// ---------------------------------------------------------------------------

TEST(KNearestNeighborsTest, K1NearestNeighbor) {
    PrintSection("K=1 Nearest Neighbor",
                 "With k=1 the prediction always matches the single closest "
                 "training example -- the simplest decision boundary.");

    std::vector<KNN::DataPoint> data = {
        {{0.0, 0.0}, 0}, {{2.0, 0.0}, 1}
    };

    KNN model(1);
    model.Fit(data);

    EXPECT_EQ(model.Predict({0.5, 0.0}), 0)
        << "Closer to (0,0) -> class 0";
    EXPECT_EQ(model.Predict({1.5, 0.0}), 1)
        << "Closer to (2,0) -> class 1";
    // Exact point
    EXPECT_EQ(model.Predict({0.0, 0.0}), 0)
        << "Exact match with training point -> class 0";

    std::cout << "  K=1 follows the single nearest training example.\n";
}

// ---------------------------------------------------------------------------
// High-dimensional (5D)
// ---------------------------------------------------------------------------

TEST(KNearestNeighborsTest, HighDimensional5D) {
    PrintSection("High-Dimensional (5D) Classification",
                 "KNN must generalize beyond 2D. Euclidean distance in 5D "
                 "still produces correct nearest-neighbor ordering.");

    // Cluster A around origin, Cluster B around (10,10,10,10,10)
    std::vector<KNN::DataPoint> data = {
        {{0.0, 0.0, 0.0, 0.0, 0.0}, 0},
        {{1.0, 0.0, 0.0, 0.0, 0.0}, 0},
        {{0.0, 1.0, 0.0, 0.0, 0.0}, 0},
        {{0.0, 0.0, 1.0, 0.0, 0.0}, 0},
        {{10.0, 10.0, 10.0, 10.0, 10.0}, 1},
        {{11.0, 10.0, 10.0, 10.0, 10.0}, 1},
        {{10.0, 11.0, 10.0, 10.0, 10.0}, 1},
        {{10.0, 10.0, 11.0, 10.0, 10.0}, 1},
    };

    KNN model(3);
    model.Fit(data);

    EXPECT_EQ(model.Predict({0.5, 0.5, 0.5, 0.0, 0.0}), 0)
        << "Near-origin 5D query -> class 0";
    EXPECT_EQ(model.Predict({10.5, 10.5, 10.5, 10.0, 10.0}), 1)
        << "Far 5D query -> class 1";

    std::cout << "  5D classification works correctly with Euclidean distance.\n";
}

// ---------------------------------------------------------------------------
// Tie-breaking consistency
// ---------------------------------------------------------------------------

TEST(KNearestNeighborsTest, TieBreakingConsistency) {
    PrintSection("Tie-Breaking Consistency",
                 "When K neighbors produce a tied vote the algorithm must "
                 "return a deterministic result across repeated calls.");

    // Two class-0, two class-1, all equidistant from query
    std::vector<KNN::DataPoint> data = {
        {{1.0, 0.0}, 0}, {{-1.0, 0.0}, 0},
        {{0.0, 1.0}, 1}, {{0.0, -1.0}, 1}
    };

    KNN model(4);  // k = 4, guaranteed tie (2 vs 2)
    model.Fit(data);

    int first = model.Predict({0.0, 0.0});
    // Repeat several times -- result must be deterministic
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(model.Predict({0.0, 0.0}), first)
            << "Tie-breaking must be deterministic on call " << i;
    }

    std::cout << "  Tied vote returned label " << first
              << " consistently across 10 calls.\n";
}

// ---------------------------------------------------------------------------
// Euclidean distance correctness
// ---------------------------------------------------------------------------

TEST(KNearestNeighborsTest, EuclideanDistanceNDimensional) {
    PrintSection("Euclidean Distance -- N-Dimensional",
                 "The distance function must correctly compute ||a-b||_2 "
                 "for vectors of any dimensionality.");

    // 2D: distance (0,0) to (3,4) = 5
    double d2 = KNN::EuclideanDistance({0.0, 0.0}, {3.0, 4.0});
    EXPECT_NEAR(d2, 5.0, 1e-10);

    // 3D: distance (1,2,3) to (4,6,3) = sqrt(9+16+0) = 5
    double d3 = KNN::EuclideanDistance({1.0, 2.0, 3.0}, {4.0, 6.0, 3.0});
    EXPECT_NEAR(d3, 5.0, 1e-10);

    // 1D: distance (0) to (7) = 7
    double d1 = KNN::EuclideanDistance({0.0}, {7.0});
    EXPECT_NEAR(d1, 7.0, 1e-10);

    // 5D: zero distance
    double d0 = KNN::EuclideanDistance({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5});
    EXPECT_NEAR(d0, 0.0, 1e-10);

    std::cout << "  2D dist=5, 3D dist=5, 1D dist=7, 5D self-dist=0\n";
}

// ---------------------------------------------------------------------------
// PredictBatch
// ---------------------------------------------------------------------------

TEST(KNearestNeighborsTest, PredictBatchMatchesSingle) {
    PrintSection("PredictBatch Matches Single Predictions",
                 "Batch prediction must return the same labels as calling "
                 "Predict individually on each query.");

    std::vector<KNN::DataPoint> data = {
        {{0.0, 0.0}, 0}, {{1.0, 0.0}, 0},
        {{10.0, 10.0}, 1}, {{11.0, 10.0}, 1}
    };

    KNN model(2);
    model.Fit(data);

    std::vector<KNN::Vector> queries = {
        {0.5, 0.0}, {10.5, 10.0}, {0.0, 0.0}
    };

    auto batch = model.PredictBatch(queries);
    ASSERT_EQ(batch.size(), queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
        EXPECT_EQ(batch[i], model.Predict(queries[i]))
            << "Batch[" << i << "] must match single Predict";
    }

    std::cout << "  Batch results match single predictions for all queries.\n";
}

// ---------------------------------------------------------------------------
// All same label
// ---------------------------------------------------------------------------

TEST(KNearestNeighborsTest, AllSameLabel) {
    PrintSection("All Same Label",
                 "When every training point shares the same label, KNN "
                 "must always predict that label regardless of query.");

    std::vector<KNN::DataPoint> data = {
        {{0.0, 0.0}, 1}, {{1.0, 1.0}, 1}, {{2.0, 2.0}, 1}
    };

    KNN model(3);
    model.Fit(data);

    EXPECT_EQ(model.Predict({100.0, 100.0}), 1);
    EXPECT_EQ(model.Predict({-50.0, -50.0}), 1);

    std::cout << "  Always predicts 1 when all training labels are 1.\n";
}
