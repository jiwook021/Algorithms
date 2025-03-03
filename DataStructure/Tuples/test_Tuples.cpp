/**
 * @file test_Tuples.cpp
 * @brief Unit tests for Tuples utilities
 */

#include "Tuples.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <vector>

/// SumMinMaxAvg returns correct statistics for a simple vector.
TEST(TuplesTest, SumMinMaxAvg) {
    std::vector<double> v = {1, 2, 3, 4, 5};
    auto [sum, mn, mx, avg] = SumMinMaxAvg(v);
    EXPECT_DOUBLE_EQ(sum, 15.0);
    EXPECT_DOUBLE_EQ(mn, 1.0);
    EXPECT_DOUBLE_EQ(mx, 5.0);
    EXPECT_DOUBLE_EQ(avg, 3.0);
}

/// SumMinMaxAvg works with a single element.
TEST(TuplesTest, SumMinMaxAvgSingle) {
    std::vector<double> v = {42.0};
    auto [sum, mn, mx, avg] = SumMinMaxAvg(v);
    EXPECT_DOUBLE_EQ(sum, 42.0);
    EXPECT_DOUBLE_EQ(mn, 42.0);
    EXPECT_DOUBLE_EQ(mx, 42.0);
    EXPECT_DOUBLE_EQ(avg, 42.0);
}

/// Zip interleaves two tuples correctly.
TEST(TuplesTest, ZipTuples) {
    auto a = std::make_tuple(1, 2);
    auto b = std::make_tuple("a", "b");
    auto result = Zip(a, b);
    EXPECT_EQ(std::get<0>(result), 1);
    EXPECT_STREQ(std::get<1>(result), "a");
    EXPECT_EQ(std::get<2>(result), 2);
    EXPECT_STREQ(std::get<3>(result), "b");
}

/// PrintArgs formats output correctly via operator<<.
TEST(TuplesTest, TuplePrint) {
    auto t = std::make_tuple(1, "hello", 3.14);
    std::ostringstream oss;
    oss << t;
    std::string output = oss.str();
    EXPECT_TRUE(output.find("1") != std::string::npos);
    EXPECT_TRUE(output.find("hello") != std::string::npos);
    EXPECT_TRUE(output.find("3.14") != std::string::npos);
    EXPECT_EQ(output.front(), '(');
    EXPECT_EQ(output.back(), ')');
}

/// Structured bindings work with tuples.
TEST(TuplesTest, StructuredBindings) {
    auto t = std::make_tuple(1, "hello", 3.14);
    auto& [a, b, c] = t;
    EXPECT_EQ(a, 1);
    EXPECT_STREQ(b, "hello");
    EXPECT_DOUBLE_EQ(c, 3.14);
}

/// std::apply invokes a callable with tuple elements.
TEST(TuplesTest, Apply) {
    auto add = [](int a, int b) { return a + b; };
    auto result = std::apply(add, std::make_tuple(3, 4));
    EXPECT_EQ(result, 7);
}
