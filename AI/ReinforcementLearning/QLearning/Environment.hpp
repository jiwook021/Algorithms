/**
 * @file Environment.hpp
 * @brief Abstract interface for reinforcement learning environments
 * @details Templated on State and Action types. Provides the standard
 *          reset / step / render contract that any RL environment must satisfy.
 */

#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <stdexcept>

/**
 * @brief Custom exception class for reinforcement learning errors
 */
class RLException : public std::runtime_error {
public:
    explicit RLException(const std::string& Message) : std::runtime_error(Message) {}
};

/**
 * @brief Abstract interface for an RL environment
 *
 * @tparam State  Type representing environment states (must be equality_comparable)
 * @tparam Action Type representing agent actions  (must be equality_comparable)
 */
template <typename State, typename Action>
class Environment {
public:
    virtual ~Environment() = default;

    /// Reset environment to initial state and return it
    virtual State Reset() = 0;

    /// Take an action; return (next_state, reward, done)
    virtual std::tuple<State, double, bool> Step(const Action& act) = 0;

    /// Get the current state
    virtual State GetCurrentState() const = 0;

    /// Check whether an action is valid from the current state
    virtual bool IsValidAction(const Action& act) const = 0;

    /// Return all valid actions from the current state
    virtual std::vector<Action> GetValidActions() const = 0;

    /// Render / visualise the environment (to stdout)
    virtual void Render() const = 0;
};
