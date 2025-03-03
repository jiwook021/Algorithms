/**
 * @file HanoiTower.cpp
 * @brief Implementation of Tower of Hanoi solver
 */

#include "HanoiTower.hpp"

namespace HanoiTower {

void Solve(int n, int src, int mid, int dest, std::vector<Move>& moves) {
    if (n == 1) {
        moves.push_back({src, dest});
        return;
    }
    Solve(n - 1, src, dest, mid, moves);
    moves.push_back({src, dest});
    Solve(n - 1, mid, src, dest, moves);
}

std::vector<Move> Solve(int n, int src, int mid, int dest) {
    std::vector<Move> moves;
    Solve(n, src, mid, dest, moves);
    return moves;
}

} // namespace HanoiTower
