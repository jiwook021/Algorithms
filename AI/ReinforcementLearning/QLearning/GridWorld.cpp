/**
 * @file GridWorld.cpp
 * @brief Implementation of State, Action helpers, and GridWorld environment
 */

#include "GridWorld.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------

bool State::operator==(const State& Other) const {
    return x == Other.x && y == Other.y;
}

std::string State::ToString() const {
    std::stringstream ss;
    ss << "(" << x << "," << y << ")";
    return ss.str();
}

// -----------------------------------------------------------------------
// Action helpers
// -----------------------------------------------------------------------

std::string ActionToString(Action act) {
    switch (act) {
        case Action::UP:    return "UP";
        case Action::RIGHT: return "RIGHT";
        case Action::DOWN:  return "DOWN";
        case Action::LEFT:  return "LEFT";
        default: throw RLException("Invalid action");
    }
}

char ActionToArrow(Action act) {
    switch (act) {
        case Action::UP:    return '^';
        case Action::RIGHT: return '>';
        case Action::DOWN:  return 'v';
        case Action::LEFT:  return '<';
        default: return '?';
    }
}

const std::vector<Action> kAllActions = {
    Action::UP, Action::RIGHT, Action::DOWN, Action::LEFT
};

// -----------------------------------------------------------------------
// GridWorld
// -----------------------------------------------------------------------

GridWorld::GridWorld(
    int width, int height, State initial, State goal,
    const std::vector<State>& obstacles
) : width_(width),
    height_(height),
    current_state_(initial),
    initial_state_(initial),
    goal_state_(goal),
    obstacles_(obstacles)
{
    if (width_ <= 0 || height_ <= 0) {
        throw RLException("Grid dimensions must be positive");
    }
    if (initial_state_.x < 0 || initial_state_.x >= width_ ||
        initial_state_.y < 0 || initial_state_.y >= height_) {
        throw RLException("Initial state outside grid boundaries");
    }
    if (goal_state_.x < 0 || goal_state_.x >= width_ ||
        goal_state_.y < 0 || goal_state_.y >= height_) {
        throw RLException("Goal state outside grid boundaries");
    }
    for (const auto& obs : obstacles_) {
        if (initial_state_ == obs || goal_state_ == obs) {
            throw RLException("Initial or goal state cannot be an obstacle");
        }
        if (obs.x < 0 || obs.x >= width_ || obs.y < 0 || obs.y >= height_) {
            throw RLException("Obstacle outside grid boundaries");
        }
    }
}

State GridWorld::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_state_ = initial_state_;
    return current_state_;
}

std::tuple<State, double, bool> GridWorld::Step(const Action& act) {
    std::lock_guard<std::mutex> lock(mutex_);

    State next = current_state_;
    switch (act) {
        case Action::UP:    next.y = std::max(0, next.y - 1);            break;
        case Action::RIGHT: next.x = std::min(width_ - 1, next.x + 1);  break;
        case Action::DOWN:  next.y = std::min(height_ - 1, next.y + 1);  break;
        case Action::LEFT:  next.x = std::max(0, next.x - 1);           break;
    }

    if (std::find(obstacles_.begin(), obstacles_.end(), next) != obstacles_.end()) {
        return {current_state_, -1.0, false};
    }

    current_state_ = next;
    bool done = (current_state_ == goal_state_);
    double reward = done ? 10.0 : -0.1;
    return {current_state_, reward, done};
}

State GridWorld::GetCurrentState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_state_;
}

bool GridWorld::IsValidAction(const Action& act) const {
    std::lock_guard<std::mutex> lock(mutex_);
    State next = current_state_;
    switch (act) {
        case Action::UP:    next.y = std::max(0, next.y - 1);            break;
        case Action::RIGHT: next.x = std::min(width_ - 1, next.x + 1);  break;
        case Action::DOWN:  next.y = std::min(height_ - 1, next.y + 1);  break;
        case Action::LEFT:  next.x = std::max(0, next.x - 1);           break;
    }
    return next.x >= 0 && next.x < width_ &&
           next.y >= 0 && next.y < height_;
}

std::vector<Action> GridWorld::GetValidActions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Action> valid;
    for (const auto& a : kAllActions) {
        State next = current_state_;
        switch (a) {
            case Action::UP:    next.y = std::max(0, next.y - 1);            break;
            case Action::RIGHT: next.x = std::min(width_ - 1, next.x + 1);  break;
            case Action::DOWN:  next.y = std::min(height_ - 1, next.y + 1);  break;
            case Action::LEFT:  next.x = std::max(0, next.x - 1);           break;
        }
        if (next.x >= 0 && next.x < width_ &&
            next.y >= 0 && next.y < height_) {
            valid.push_back(a);
        }
    }
    return valid;
}

void GridWorld::Render() const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Column headers
    std::cout << "    ";
    for (int c = 0; c < width_; ++c) {
        std::cout << " " << c << "  ";
    }
    std::cout << "\n";

    std::string border = "  +";
    for (int c = 0; c < width_; ++c) border += "---+";

    for (int r = 0; r < height_; ++r) {
        std::cout << border << "\n";
        std::cout << r << " |";
        for (int c = 0; c < width_; ++c) {
            State s{c, r};
            char ch = '.';
            if (current_state_ == goal_state_ && s == current_state_) ch = '*';
            else if (s == current_state_) ch = 'A';
            else if (s == goal_state_) ch = 'G';
            else if (std::find(obstacles_.begin(), obstacles_.end(), s) != obstacles_.end()) ch = '#';
            std::cout << " " << ch << " |";
        }
        std::cout << "\n";
    }
    std::cout << border << "\n";
}

int GridWorld::GetWidth() const { return width_; }
int GridWorld::GetHeight() const { return height_; }
State GridWorld::GetGoalState() const { return goal_state_; }

bool GridWorld::IsObstacle(const State& s) const {
    return std::find(obstacles_.begin(), obstacles_.end(), s) != obstacles_.end();
}

std::shared_ptr<GridWorld> CreateSimpleGridWorld() {
    int width  = 5;
    int height = 5;
    State initial{0, 0};
    State goal{4, 4};
    std::vector<State> obstacles = {
        {1, 1}, {1, 2}, {1, 3},
        {3, 0}, {3, 1}, {3, 2}
    };
    return std::make_shared<GridWorld>(width, height, initial, goal, obstacles);
}
