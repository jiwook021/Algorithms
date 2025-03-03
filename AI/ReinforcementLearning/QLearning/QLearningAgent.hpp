/**
 * @file QLearningAgent.hpp
 * @brief Q-Learning agent templated on State and Action types
 * @details Epsilon-greedy exploration with Bellman-equation updates.
 *          Requires State and Action to be equality_comparable.
 */

#pragma once

#include "Environment.hpp"

#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <string>
#include <memory>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <concepts>
#include <limits>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <sstream>

// -----------------------------------------------------------------------
// RLAgent (abstract base)
// -----------------------------------------------------------------------

/**
 * @brief Abstract interface for reinforcement learning agents
 */
template <typename S, typename A>
class RLAgent {
public:
    virtual ~RLAgent() = default;

    virtual A ChooseAction(const S& state) = 0;
    virtual void Update(const S& state, const A& action, double reward,
                        const S& next_state, bool done) = 0;
    virtual void SavePolicy(const std::string& filename) const = 0;
    virtual void LoadPolicy(const std::string& filename) = 0;
};

// -----------------------------------------------------------------------
// QLearningAgent
// -----------------------------------------------------------------------

/**
 * @brief Q-Learning agent implementation
 *
 * Model-free RL that learns a state-action value function via the
 * Bellman equation with epsilon-greedy exploration.
 *
 * @tparam State  Must be std::equality_comparable and hashable
 * @tparam Action Must be std::equality_comparable and hashable
 */
template <typename State, typename Action>
    requires std::equality_comparable<State> && std::equality_comparable<Action>
class QLearningAgent : public RLAgent<State, Action> {
private:
    // Q-Table: maps state -> (action -> Q-value)
    std::unordered_map<State, std::unordered_map<Action, double>> q_table_;

    double learning_rate_;
    double discount_factor_;
    double exploration_rate_;
    double exploration_decay_;
    double min_exploration_rate_;

    std::shared_ptr<std::mt19937> rng_;
    mutable std::mutex mutex_;

    // All actions the agent knows about
    std::vector<Action> all_actions_;

public:
    /**
     * @brief Construct a new QLearningAgent
     *
     * @param all_actions    Every action the agent may take
     * @param learning_rate  Alpha  -- how quickly new info replaces old (0,1]
     * @param discount_factor Gamma -- importance of future rewards [0,1)
     * @param exploration_rate Epsilon -- probability of random action [0,1]
     * @param exploration_decay Multiplicative decay per update (0,1]
     * @param min_exploration_rate Floor for epsilon [0, exploration_rate]
     * @param seed           RNG seed for reproducibility
     *
     * @throws RLException on invalid parameters
     */
    QLearningAgent(
        const std::vector<Action>& all_actions,
        double learning_rate       = 0.1,
        double discount_factor     = 0.99,
        double exploration_rate    = 1.0,
        double exploration_decay   = 0.995,
        double min_exploration_rate = 0.01,
        unsigned int seed          = std::random_device{}()
    ) : learning_rate_(learning_rate),
        discount_factor_(discount_factor),
        exploration_rate_(exploration_rate),
        exploration_decay_(exploration_decay),
        min_exploration_rate_(min_exploration_rate),
        rng_(std::make_shared<std::mt19937>(seed)),
        all_actions_(all_actions)
    {
        if (learning_rate_ <= 0 || learning_rate_ > 1) {
            throw RLException("Learning rate must be in range (0, 1]");
        }
        if (discount_factor_ < 0 || discount_factor_ >= 1) {
            throw RLException("Discount factor must be in range [0, 1)");
        }
        if (exploration_rate_ < 0 || exploration_rate_ > 1) {
            throw RLException("Exploration rate must be in range [0, 1]");
        }
        if (exploration_decay_ <= 0 || exploration_decay_ > 1) {
            throw RLException("Exploration decay must be in range (0, 1]");
        }
        if (min_exploration_rate_ < 0 || min_exploration_rate_ > exploration_rate_) {
            throw RLException("Min exploration rate must be in range [0, explorationRate]");
        }
        if (all_actions_.empty()) {
            throw RLException("Action set must not be empty");
        }
    }

    // ----- RLAgent interface -----

    /**
     * @brief Choose an action via epsilon-greedy policy
     */
    Action ChooseAction(const State& state) override {
        std::lock_guard<std::mutex> lock(mutex_);

        std::uniform_real_distribution<double> coin(0.0, 1.0);
        if (coin(*rng_) < exploration_rate_) {
            std::uniform_int_distribution<size_t> idx(0, all_actions_.size() - 1);
            return all_actions_[idx(*rng_)];
        }

        return GetBestActionLocked(state, all_actions_);
    }

    /**
     * @brief Update Q-values: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
     */
    void Update(
        const State& state,
        const Action& action,
        double reward,
        const State& next_state,
        bool done
    ) override {
        std::lock_guard<std::mutex> lock(mutex_);

        // Initialise if missing
        if (q_table_.find(state) == q_table_.end() ||
            q_table_[state].find(action) == q_table_[state].end()) {
            q_table_[state][action] = 0.0;
        }

        double max_next_q = 0.0;
        if (!done) {
            max_next_q = GetMaxQValueLocked(next_state);
        }

        double current_q = q_table_[state][action];
        double new_q = current_q + learning_rate_ * (reward + discount_factor_ * max_next_q - current_q);
        q_table_[state][action] = new_q;

        // Decay exploration
        exploration_rate_ = std::max(min_exploration_rate_,
                                     exploration_rate_ * exploration_decay_);
    }

    /**
     * @brief Save Q-table to a file (CSV: state_fields, action_int, value)
     *
     * NOTE: For generic State/Action this writes nothing. Specialise or
     *       override for custom serialisation.
     */
    void SavePolicy(const std::string& /*filename*/) const override {
        // Generic stub -- specialisations for concrete types can override.
    }

    /**
     * @brief Load Q-table from a file
     */
    void LoadPolicy(const std::string& /*filename*/) override {
        // Generic stub
    }

    // ----- Public helpers -----

    double GetExplorationRate() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return exploration_rate_;
    }

    /**
     * @brief Thread-safe public wrapper for getting best action
     */
    Action GetBestAction(const State& state, const std::vector<Action>& actions) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return GetBestActionLocked(state, actions);
    }

    /**
     * @brief Direct (non-locking) access to all actions
     */
    const std::vector<Action>& GetAllActions() const {
        return all_actions_;
    }

    /**
     * @brief Retrieve the Q-value for a (state, action) pair (0 if unseen)
     */
    double GetQValue(const State& state, const Action& action) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto sit = q_table_.find(state);
        if (sit == q_table_.end()) return 0.0;
        auto ait = sit->second.find(action);
        if (ait == sit->second.end()) return 0.0;
        return ait->second;
    }

private:
    /**
     * @brief Get best action (must be called while mutex_ is held)
     */
    Action GetBestActionLocked(const State& state, const std::vector<Action>& actions) const {
        auto it = q_table_.find(state);
        if (it == q_table_.end()) {
            // State unseen: pick randomly
            std::uniform_int_distribution<size_t> dist(0, actions.size() - 1);
            return actions[dist(*rng_)];
        }

        const auto& state_q = it->second;
        double max_q = std::numeric_limits<double>::lowest();
        std::vector<Action> best;

        for (const auto& a : actions) {
            double qv = 0.0;
            auto ait = state_q.find(a);
            if (ait != state_q.end()) {
                qv = ait->second;
            }
            if (qv > max_q) {
                max_q = qv;
                best.clear();
                best.push_back(a);
            } else if (qv == max_q) {
                best.push_back(a);
            }
        }

        if (best.size() > 1) {
            std::uniform_int_distribution<size_t> dist(0, best.size() - 1);
            return best[dist(*rng_)];
        }
        return best[0];
    }

    /**
     * @brief Max Q-value over all actions for a state (must hold mutex_)
     */
    double GetMaxQValueLocked(const State& state) const {
        auto it = q_table_.find(state);
        if (it == q_table_.end()) return 0.0;
        const auto& state_q = it->second;
        if (state_q.empty()) return 0.0;

        double max_q = std::numeric_limits<double>::lowest();
        for (const auto& [act, qv] : state_q) {
            max_q = std::max(max_q, qv);
        }
        return max_q;
    }
};

// -----------------------------------------------------------------------
// Training / testing free functions
// -----------------------------------------------------------------------

/**
 * @brief Train a QLearningAgent in an Environment
 *
 * @return Average reward per episode
 */
template <typename State, typename Action>
double TrainAgent(
    std::shared_ptr<Environment<State, Action>> env,
    std::shared_ptr<QLearningAgent<State, Action>> agent,
    int episodes,
    int max_steps,
    bool verbose = false
) {
    double total_reward = 0.0;

    for (int ep = 0; ep < episodes; ++ep) {
        State st = env->Reset();
        double episode_reward = 0.0;
        int steps = 0;
        bool done = false;

        while (!done && steps < max_steps) {
            Action act = agent->ChooseAction(st);
            auto [next_state, reward, is_done] = env->Step(act);
            agent->Update(st, act, reward, next_state, is_done);

            st = next_state;
            episode_reward += reward;
            done = is_done;
            ++steps;
        }

        total_reward += episode_reward;

        if (verbose && ep % 100 == 0) {
            std::cout << "  Ep " << std::setw(5) << (ep + 1) << "/" << episodes
                      << " | Steps: " << std::setw(3) << steps
                      << " | Reward: " << std::setw(7) << std::fixed << std::setprecision(2) << episode_reward
                      << " | Explore: " << std::fixed << std::setprecision(4)
                      << agent->GetExplorationRate() << "\n";
        }
    }

    return total_reward / episodes;
}

/**
 * @brief Test a trained agent
 *
 * @return Average reward per episode
 */
template <typename State, typename Action>
double TestAgent(
    std::shared_ptr<Environment<State, Action>> env,
    std::shared_ptr<QLearningAgent<State, Action>> agent,
    int episodes,
    int max_steps,
    bool render = true
) {
    double total_reward = 0.0;
    int total_steps = 0;
    int success_count = 0;

    const auto& actions = agent->GetAllActions();

    for (int ep = 0; ep < episodes; ++ep) {
        State st = env->Reset();
        double episode_reward = 0.0;
        int steps = 0;
        bool done = false;

        if (render) {
            std::cout << "\nTest Episode " << (ep + 1) << " Start:\n";
            env->Render();
        }

        while (!done && steps < max_steps) {
            Action act = agent->GetBestAction(st, actions);
            auto [next_state, reward, is_done] = env->Step(act);

            st = next_state;
            episode_reward += reward;
            done = is_done;
            ++steps;

            if (render) {
                std::cout << "Step " << steps << ": Reward = "
                          << std::fixed << std::setprecision(2) << reward << "\n";
                env->Render();
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
        }

        total_reward += episode_reward;
        total_steps += steps;
        if (done) ++success_count;

        if (render) {
            std::cout << "Episode " << (ep + 1) << " finished. Steps: " << steps
                      << ", Reward: " << std::fixed << std::setprecision(2) << episode_reward
                      << ", Success: " << (done ? "Yes" : "No") << "\n";
        }
    }

    double avg_reward  = total_reward / episodes;
    double avg_steps   = static_cast<double>(total_steps) / episodes;
    double success_pct = static_cast<double>(success_count) / episodes * 100.0;

    std::cout << "\nTest Results:\n";
    std::cout << "Average Reward: "  << std::fixed << std::setprecision(2) << avg_reward  << "\n";
    std::cout << "Average Steps: "   << std::fixed << std::setprecision(2) << avg_steps   << "\n";
    std::cout << "Success Rate: "    << std::fixed << std::setprecision(2) << success_pct << "%\n";

    return avg_reward;
}
