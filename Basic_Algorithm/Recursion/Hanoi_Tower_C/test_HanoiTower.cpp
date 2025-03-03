/**
 * @file test_HanoiTower.cpp
 * @brief Unit tests for HanoiTower
 */

#include "HanoiTower.hpp"
#include <gtest/gtest.h>

using namespace HanoiTower;

TEST(HanoiTower, OneDisc) {
    auto moves = Solve(1, 1, 2, 3);
    ASSERT_EQ(moves.size(), 1u);
    EXPECT_EQ(moves[0].From, 1);
    EXPECT_EQ(moves[0].To, 3);
}

TEST(HanoiTower, TwoDiscs) {
    auto moves = Solve(2, 1, 2, 3);
    EXPECT_EQ(moves.size(), 3u);
}

TEST(HanoiTower, ThreeDiscs) {
    auto moves = Solve(3, 1, 2, 3);
    EXPECT_EQ(moves.size(), 7u);
}

TEST(HanoiTower, FourDiscs) {
    auto moves = Solve(4, 1, 2, 3);
    EXPECT_EQ(moves.size(), 15u);
}

TEST(HanoiTower, MoveCountFormula) {
    for (int n = 1; n <= 10; ++n) {
        auto moves = Solve(n, 1, 2, 3);
        EXPECT_EQ(moves.size(), (1u << n) - 1);
    }
}

TEST(HanoiTower, FirstAndLastMove) {
    auto moves = Solve(3, 1, 2, 3);
    // First move should be from src
    EXPECT_EQ(moves.front().From, 1);
    // Last move should be to dest
    EXPECT_EQ(moves.back().To, 3);
}
