/**
 * @file test_FeatureDeployment.cpp
 * @brief Unit tests for FeatureDeployment
 */

#include "FeatureDeployment.hpp"
#include <gtest/gtest.h>

using namespace FeatureDeployment;

TEST(FeatureDeployment, Example1) {
    auto result = Solution({93, 30, 55}, {1, 30, 5});
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 1);
}

TEST(FeatureDeployment, Example2) {
    auto result = Solution({95, 90, 99, 99, 80, 99}, {1, 1, 1, 1, 1, 1});
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 2);
}

TEST(FeatureDeployment, AllSameSpeed) {
    auto result = Solution({50, 50, 50}, {10, 10, 10});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 3);
}

TEST(FeatureDeployment, SingleFeature) {
    auto result = Solution({0}, {1});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 1);
}

TEST(FeatureDeployment, AllDifferentDays) {
    auto result = Solution({0, 0, 0}, {100, 50, 25});
    // Days: 1, 2, 4 -- each deploys separately
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 1);
    EXPECT_EQ(result[2], 1);
}

TEST(FeatureDeployment, EmptyInput) {
    auto result = Solution({}, {});
    EXPECT_TRUE(result.empty());
}
