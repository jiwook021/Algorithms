/**
 * @file SelfTuningPIDController.hpp
 * @brief Reinforcement learning-based self-tuning PID controller
 * @details Top-level include that aggregates all module components:
 *          PIDController, NeuralNetwork, Environment, RLAgent, and
 *          SelfTuningPIDController.
 *
 * Time Complexity: O(E * T * L * N^2) for training, where E = episodes,
 *                  T = steps per episode, L = network layers, N = neurons
 * Space Complexity: O(L * N^2) for network weights
 */

#pragma once

#include "PIDController.hpp"
#include "NeuralNetwork.hpp"
#include "Environment.hpp"
#include "RLAgent.hpp"

#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <vector>

/**
 * @class SelfTuningPIDController
 * @brief Orchestrates RL-based PID gain optimization
 *
 * Combines PIDController, RLAgent, and Environment to automatically
 * find optimal PID parameters via Q-learning or grid search.
 */
class SelfTuningPIDController {
public:
    /**
     * @brief Construct a self-tuning PID system
     * @param env The process environment
     * @param pidController The PID controller to tune
     * @param hiddenSize Hidden layer size for the RL agent's network
     */
    SelfTuningPIDController(std::shared_ptr<Environment> env,
                             std::shared_ptr<PIDController> pidController,
                             size_t hiddenSize = 24);

    /**
     * @brief Train the RL agent
     * @param numEpisodes Number of training episodes
     * @param targetSyncFrequency Episodes between target network syncs
     * @param maxStepsPerEpisode Maximum steps per episode
     */
    void Train(int numEpisodes, int targetSyncFrequency = 10, int maxStepsPerEpisode = 100);

    /**
     * @brief Use grid search to find optimal PID parameters
     * @param numTests Steps per parameter set evaluation
     */
    void SimpleTuning(int numTests = 100);

    /** @brief Request training to stop */
    void StopTraining();

    /**
     * @brief Run the controller with fixed parameters
     * @param numSteps Number of simulation steps
     * @param setpoint Target setpoint
     * @return History of [time, setpoint, processValue, controlSignal]
     */
    std::vector<std::vector<double>> Run(int numSteps, double setpoint);

    /** @brief Save the RL agent model */
    void SaveModel(const std::string& filename);
    /** @brief Load an RL agent model */
    void LoadModel(const std::string& filename);

private:
    std::shared_ptr<Environment> env_;
    std::shared_ptr<PIDController> pidController_;
    RLAgent rlAgent_;
    std::atomic<bool> isTraining_;
};
