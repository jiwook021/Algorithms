/**
 * @file HanoiTower.hpp
 * @brief Tower of Hanoi recursive solution
 * @details Solves the classic Tower of Hanoi problem by recording moves
 *          rather than printing them, enabling testability.
 */

#pragma once

#include <vector>

namespace HanoiTower {

struct Move {
    int From;
    int To;
};

/**
 * @brief Solve Tower of Hanoi, recording moves.
 * @param n     Number of disks
 * @param src   Source peg identifier
 * @param mid   Middle (auxiliary) peg identifier
 * @param dest  Destination peg identifier
 * @param moves Output vector of moves
 */
void Solve(int n, int src, int mid, int dest, std::vector<Move>& moves);

/**
 * @brief Convenience wrapper that returns the move sequence.
 */
std::vector<Move> Solve(int n, int src, int mid, int dest);

} // namespace HanoiTower
