/**
 * @file test_HierarchicalClustering.cpp
 * @brief Unit tests for HierarchicalClustering.
 */

#include <cmath>
#include <iostream>

#include "HierarchicalClustering.hpp"

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_TRUE(cond, msg)                                       \
    do {                                                             \
        if (!(cond)) {                                               \
            std::cerr << "FAIL: " << msg << "\n";                    \
            testsFailed++;                                           \
        } else {                                                     \
            std::cout << "PASS: " << msg << "\n";                    \
            testsPassed++;                                           \
        }                                                            \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg) ASSERT_TRUE(std::fabs((a)-(b)) < (tol), msg)
#define ASSERT_EQ(a, b, msg) ASSERT_TRUE((a) == (b), msg)

/**
 * @brief Euclidean distance: (0,0) to (3,4) should be 5.
 */
void TestDistance() {
    ASSERT_NEAR(HierarchicalClustering::Distance({0, 0}, {3, 4}), 5.0, 1e-10,
                "Distance (0,0)-(3,4) = 5");
}

/**
 * @brief Distance from a point to itself is 0.
 */
void TestDistanceSelf() {
    ASSERT_NEAR(HierarchicalClustering::Distance({2, 3}, {2, 3}), 0.0, 1e-10,
                "Distance from point to itself is 0");
}

/**
 * @brief Single-linkage cluster distance.
 */
void TestClusterDistance() {
    Cluster c1{{{0, 0}, {1, 0}}};
    Cluster c2{{{5, 0}, {6, 0}}};
    ASSERT_NEAR(HierarchicalClustering::ClusterDistance(c1, c2), 4.0, 1e-10,
                "ClusterDistance between {(0,0),(1,0)} and {(5,0),(6,0)} = 4");
}

/**
 * @brief FindClosestClusters picks the nearest pair.
 */
void TestFindClosest() {
    std::vector<Cluster> clusters = {{{{0, 0}}}, {{{1, 0}}}, {{{10, 0}}}};
    auto [a, b] = HierarchicalClustering::FindClosestClusters(clusters);
    ASSERT_EQ(a, 0, "FindClosest: first index is 0");
    ASSERT_EQ(b, 1, "FindClosest: second index is 1");
}

/**
 * @brief MergeClusters combines points from both clusters.
 */
void TestMerge() {
    Cluster c1{{{0, 0}}};
    Cluster c2{{{1, 1}}};
    auto merged = HierarchicalClustering::MergeClusters(c1, c2);
    ASSERT_EQ(merged.Points.size(), 2u, "Merged cluster has 2 points");
}

/**
 * @brief Fit reduces to target number of clusters.
 */
void TestFitTargetClusters() {
    std::vector<Point> data = {{0, 0}, {1, 0}, {10, 0}, {11, 0}};
    auto clusters = HierarchicalClustering::Fit(data, 2);
    ASSERT_EQ(static_cast<int>(clusters.size()), 2,
              "Fit with target=2 produces 2 clusters");
}

/**
 * @brief Fit to 1 cluster puts all points together.
 */
void TestFitSingleCluster() {
    std::vector<Point> data = {{0, 0}, {1, 1}, {2, 2}};
    auto clusters = HierarchicalClustering::Fit(data, 1);
    ASSERT_EQ(static_cast<int>(clusters.size()), 1,
              "Fit with target=1 produces 1 cluster");
    ASSERT_EQ(clusters[0].Points.size(), 3u,
              "Single cluster contains all 3 points");
}

int main() {
    std::cout << "=== HierarchicalClustering Unit Tests ===\n\n";

    TestDistance();
    TestDistanceSelf();
    TestClusterDistance();
    TestFindClosest();
    TestMerge();
    TestFitTargetClusters();
    TestFitSingleCluster();

    std::cout << "\n=== Results: " << testsPassed << " passed, "
              << testsFailed << " failed ===\n";
    return testsFailed > 0 ? 1 : 0;
}
