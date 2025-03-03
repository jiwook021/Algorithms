/**
 * @file RLAgent.hpp
 * @brief Q-learning agent for PID parameter optimization
 * @details Uses a neural network to approximate Q-values and an epsilon-greedy
 *          exploration strategy with decaying epsilon.
 *
 * Time Complexity: O(stateSize * hiddenSize + hiddenSize * actionSize) per action/learn
 * Space Complexity: O(stateSize * hiddenSize + hiddenSize * actionSize) for network weights
 */

#pragma once

#include "NeuralNetwork.hpp"
#include "Environment.hpp"
#include "PIDController.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

/**
 * @class RLAgent
 * @brief Q-learning agent for PID parameter optimization
 *
 * Uses a neural network to approximate Q-values and an epsilon-greedy
 * exploration strategy with decaying epsilon.
 */
class RLAgent {
public:
    /**
     * @brief Construct an RL agent
     * @param stateSize Dimension of the state vector
     * @param actionSize Number of PID parameters to tune
     * @param hiddenSize Hidden layer size for the Q-network
     * @param learningRate Neural network learning rate
     * @param gamma Discount factor for future rewards
     * @param epsilon Initial exploration rate
     * @param epsilonDecay Multiplicative decay per episode
     * @param epsilonMin Minimum exploration rate
     */
    RLAgent(size_t stateSize, size_t actionSize, size_t hiddenSize = 24,
            double learningRate = 0.0001, double gamma = 0.99,
            double epsilon = 1.0, double epsilonDecay = 0.995, double epsilonMin = 0.01);

    /**
     * @brief Select an action using epsilon-greedy policy
     * @param state Current state vector
     * @return Action vector (PID parameters [kp, ki, kd])
     */
    std::vector<double> SelectAction(const std::vector<double>& state);

    /**
     * @brief Update Q-network from a single experience
     * @param state Current state
     * @param action Action taken
     * @param reward Reward received
     * @param nextState Next state
     * @param done Whether episode ended
     * @return Training loss
     */
    double Learn(const std::vector<double>& state, const std::vector<double>& action,
                 double reward, const std::vector<double>& nextState, bool done);

    /** @brief Copy Q-network weights to target network */
    void SyncTargetNetwork();

    /** @brief Save Q-network model */
    void SaveModel(const std::string& filename);

    /**
     * @brief Load Q-network model
     * @throws std::runtime_error if file cannot be opened
     */
    void LoadModel(const std::string& filename);

    /** @return Current exploration rate */
    double GetEpsilon() const;

    /**
     * @brief Grid search for optimal PID parameters
     * @param env Environment to evaluate in
     * @param numTests Steps per parameter combination
     * @return Best [kp, ki, kd] found
     *
     * Time Complexity: O(|Kp| * |Ki| * |Kd| * numTests)
     */
    std::vector<double> GridSearch(std::shared_ptr<Environment> env, int numTests = 100);

private:
    size_t stateSize_, actionSize_;
    NeuralNetwork qNetwork_, targetNetwork_;
    double gamma_, epsilon_, epsilonDecay_, epsilonMin_;
    std::mt19937 gen_;
    std::vector<std::pair<double, double>> actionRanges_;
};
