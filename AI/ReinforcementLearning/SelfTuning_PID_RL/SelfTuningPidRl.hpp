/**
 * @file SelfTuningPidRl.hpp
 * @brief Self-tuning PID controller via reinforcement learning
 * @details Uses a neural network to approximate the Q-function and tunes PID
 *          gains (Kp, Ki, Kd) to minimize tracking error on a first-order process.
 */
#pragma once

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <memory>
#include <mutex>
#include <atomic>
#include <tuple>
#include <algorithm>
#include <limits>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <functional>

namespace Rl {

/**
 * @brief Traditional PID controller with anti-windup.
 */
class PidController {
public:
    PidController(double kp = 1.0, double ki = 0.0, double kd = 0.0, double dt = 0.01);
    void UpdateParameters(double kp, double ki, double kd);
    double Compute(double error);
    void Reset();
    std::vector<double> GetParameters() const;

private:
    double Kp_, Ki_, Kd_, Dt_;
    double PreviousError_, Integral_;
    mutable std::mutex Mutex_;
};

/**
 * @brief Simple feed-forward neural network (1 hidden layer, ReLU/linear).
 */
class NeuralNetwork {
public:
    NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize,
                  double learningRate = 0.001);

    std::vector<double> Forward(const std::vector<double>& input);
    double Train(const std::vector<double>& input, const std::vector<double>& target);
    void SaveModel(const std::string& filename);
    void LoadModel(const std::string& filename);

private:
    size_t InputSize_, HiddenSize_, OutputSize_;
    double LearningRate_;
    std::vector<std::vector<double>> W1_, W2_;
    std::vector<double> B1_, B2_;
    std::vector<double> Hidden_;
    mutable std::mutex Mutex_;
};

/**
 * @brief Interface for the simulated process environment.
 */
class Environment {
public:
    virtual ~Environment() = default;
    virtual std::vector<double> Reset() = 0;
    virtual std::tuple<std::vector<double>, double, bool>
        Step(const std::vector<double>& action, double controlSignal) = 0;
    virtual double GetSetpoint() const = 0;
    virtual double GetProcessValue() const = 0;
};

/**
 * @brief First-order process simulation with optional disturbance.
 */
class SimpleProcessEnvironment : public Environment {
public:
    SimpleProcessEnvironment(double timeConstant = 1.0, double gain = 1.0,
                              double dt = 0.01, double disturbanceStdDev = 0.0);

    std::vector<double> Reset() override;
    std::tuple<std::vector<double>, double, bool>
        Step(const std::vector<double>& action, double controlSignal) override;
    double GetSetpoint() const override;
    double GetProcessValue() const override;
    void SetSetpoint(double setpoint);
    void SetMaxSteps(int maxSteps);

private:
    double TimeConstant_, Gain_, Dt_, DisturbanceStdDev_;
    double ProcessValue_, Setpoint_;
    int StepCount_, MaxSteps_;
    std::mt19937 Gen_;
    std::normal_distribution<double> Dist_;
};

/**
 * @brief RL agent that tunes PID parameters via DQN.
 */
class RlAgent {
public:
    RlAgent(size_t stateSize, size_t actionSize, size_t hiddenSize = 24,
            double learningRate = 0.0001, double gamma = 0.99,
            double epsilon = 1.0, double epsilonDecay = 0.995,
            double epsilonMin = 0.01);

    std::vector<double> SelectAction(const std::vector<double>& state);
    double Learn(const std::vector<double>& state, const std::vector<double>& action,
                 double reward, const std::vector<double>& nextState, bool done);
    void SyncTargetNetwork();
    void SaveModel(const std::string& filename);
    void LoadModel(const std::string& filename);
    double GetEpsilon() const;

    std::vector<double> GridSearch(std::shared_ptr<Environment> env, int numTests = 100);

private:
    size_t StateSize_, ActionSize_;
    NeuralNetwork QNetwork_, TargetNetwork_;
    double Gamma_, Epsilon_, EpsilonDecay_, EpsilonMin_;
    std::mt19937 Gen_;
    std::vector<std::pair<double, double>> ActionRanges_;
};

/**
 * @brief Self-tuning PID controller that uses RL for parameter optimization.
 */
class SelfTuningPidController {
public:
    SelfTuningPidController(std::shared_ptr<Environment> env,
                             std::shared_ptr<PidController> pid,
                             size_t hiddenSize = 24);

    void Train(int numEpisodes, int targetSyncFreq = 10, bool verbose = true);
    void SimpleTuning(int numTests = 100);
    std::vector<std::vector<double>> Run(int numSteps, double setpoint);
    void SaveModel(const std::string& filename);
    void LoadModel(const std::string& filename);
    void StopTraining();

private:
    std::shared_ptr<Environment> Env_;
    std::shared_ptr<PidController> Pid_;
    RlAgent Agent_;
    std::atomic<bool> IsTraining_;
};

} // namespace Rl
