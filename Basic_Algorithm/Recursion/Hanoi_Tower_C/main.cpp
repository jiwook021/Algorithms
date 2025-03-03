/**
 * @file main.cpp
 * @brief Driver for HanoiTower
 */

#include "HanoiTower.hpp"
#include <cstdio>

int main() {
    auto moves = HanoiTower::Solve(3, 'A', 'B', 'C');

    for (const auto& m : moves) {
        printf("Move disk from %c to %c\n", m.From, m.To);
    }

    return 0;
}
