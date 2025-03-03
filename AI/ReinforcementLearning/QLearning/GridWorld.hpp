/**
 * @file GridWorld.hpp
 * @brief Declarations for State, Action, and the GridWorld environment
 */

#pragma once

#include "Environment.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

// -----------------------------------------------------------------------
// State — a position (x, y) on the grid
// -----------------------------------------------------------------------

struct State {
    int x;
    int y;

    bool operator==(const State& Other) const;
    std::string ToString() const;
};

// Hash specialization: required so State can be used as a key in
// std::unordered_map (which the Q-table uses: state -> action -> value).
// Combines hash(x) and hash(y) using XOR with a bit-shift so that
// (a,b) and (b,a) produce different hashes.
// Must live in the header — the compiler needs it wherever unordered_map<State>
// is instantiated.
namespace std {
template <>
struct hash<State> {
    size_t operator()(const State& st) const {
        return hash<int>()(st.x) ^ (hash<int>()(st.y) << 1);
    }
};
} // namespace std

// -----------------------------------------------------------------------
// Action — the four movement directions
// -----------------------------------------------------------------------

enum class Action { UP, RIGHT, DOWN, LEFT };

/// Convert Action to string (e.g. "RIGHT")
std::string ActionToString(Action act);

/// Convert Action to arrow character for policy display
char ActionToArrow(Action act);

/// All possible actions
extern const std::vector<Action> kAllActions;

// Hash specialization: lets Action be used as an unordered_map key.
// Casts the enum to int and hashes that.
namespace std {
template <>
struct hash<Action> {
    size_t operator()(const Action& act) const {
        return hash<int>()(static_cast<int>(act));
    }
};
} // namespace std

// -----------------------------------------------------------------------
// GridWorld — 2D grid environment with obstacles
// -----------------------------------------------------------------------

class GridWorld : public Environment<State, Action> {
private:
    int width_;
    int height_;
    State current_state_;
    State initial_state_;
    State goal_state_;
    std::vector<State> obstacles_;
    mutable std::mutex mutex_;

public:
    GridWorld(int width, int height, State initial, State goal,
             const std::vector<State>& obstacles);

    GridWorld(const GridWorld&) = delete;
    GridWorld& operator=(const GridWorld&) = delete;
    GridWorld(GridWorld&&) = default;
    GridWorld& operator=(GridWorld&&) = default;

    // Environment interface
    State Reset() override;
    std::tuple<State, double, bool> Step(const Action& act) override;
    State GetCurrentState() const override;
    bool IsValidAction(const Action& act) const override;
    std::vector<Action> GetValidActions() const override;
    void Render() const override;

    // Accessors for demo / visualization
    int GetWidth() const;
    int GetHeight() const;
    State GetGoalState() const;
    bool IsObstacle(const State& s) const;
};

/// Factory: create a 5x5 grid world with obstacles
std::shared_ptr<GridWorld> CreateSimpleGridWorld();
