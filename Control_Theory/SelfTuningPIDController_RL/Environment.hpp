/**
 * @file Environment.hpp
 * @brief Abstract RL environment interface and first-order process simulation
 * @details Defines the Environment interface and SimpleProcessEnvironment
 *          which simulates a first-order process for PID controller testing.
 *
 * Time Complexity: O(1) per Step call
 * Space Complexity: O(1)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

/**
 * @class Environment
 * @brief Abstract interface for RL environments
 */
class Environment {
public:
    virtual ~Environment() = default;

    /**
     * @brief Reset environment to initial state
     * @return Initial observation vector
     */
    virtual std::vector<double> Reset() = 0;

    /**
     * @brief Take one step in the environment
     * @param action Action vector (PID parameters)
     * @param controlSignal Control signal from PID
     * @return Tuple of (nextState, reward, done)
     */
    virtual std::tuple<std::vector<double>, double, bool> Step(
        const std::vector<double>& action, double controlSignal) = 0;

    /** @return Current setpoint */
    virtual double GetSetpoint() const = 0;
    /** @return Current process value */
    virtual double GetProcessValue() const = 0;
};

/**
 * @class SimpleProcessEnvironment
 * @brief First-order process simulation for PID controller testing
 *
 * Simulates: dPV/dt = (gain * control - PV) / timeConstant
 * with optional Gaussian disturbances.
 */
class SimpleProcessEnvironment : public Environment {
public:
    /**
     * @brief Construct the environment
     * @param timeConstant System time constant
     * @param gain System gain
     * @param dt Time step
     * @param disturbanceStdDev Standard deviation of random disturbances
     */
    SimpleProcessEnvironment(double timeConstant = 1.0, double gain = 1.0,
                              double dt = 0.01, double disturbanceStdDev = 0.0);

    /**
     * @brief Reset to initial state
     * @return Initial state [error, errorDerivative, previousControl]
     */
    std::vector<double> Reset() override;

    /**
     * @brief Advance the simulation by one step
     * @param action PID parameters [kp, ki, kd]
     * @param controlSignal Control output from PID
     * @return Tuple of (nextState, reward, done)
     */
    std::tuple<std::vector<double>, double, bool> Step(
        const std::vector<double>& action, double controlSignal) override;

    double GetSetpoint() const override;
    double GetProcessValue() const override;

    /** @brief Set a new setpoint (clamped to [-5, 5]) */
    void SetSetpoint(double setpoint);

    /** @brief Set the maximum steps per episode */
    void SetMaxSteps(int maxSteps);

private:
    double timeConstant_, gain_, dt_, disturbanceStdDev_;
    double processValue_, setpoint_;
    int stepCount_, maxSteps_;
    std::mt19937 gen_;
    std::normal_distribution<double> dist_;
};
