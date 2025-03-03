/**
 * @file PIDController.hpp
 * @brief Traditional PID controller with thread-safe parameter updates
 * @details Implements proportional, integral, and derivative control with
 *          anti-windup and derivative filtering. Parameters can be updated
 *          at runtime by the RL agent.
 *
 * Time Complexity: O(1) per Compute call
 * Space Complexity: O(1)
 */

#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

/**
 * @class PIDController
 * @brief Traditional PID controller with thread-safe parameter updates
 *
 * Implements proportional, integral, and derivative control with anti-windup
 * and derivative filtering. Parameters can be updated at runtime by the RL agent.
 */
class PIDController {
public:
    /**
     * @brief Construct a PIDController
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param kd Derivative gain
     * @param dt Time step for integration/differentiation
     */
    PIDController(double kp = 1.0, double ki = 0.0, double kd = 0.0, double dt = 0.01);

    /**
     * @brief Update PID gains (thread-safe)
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param kd Derivative gain
     */
    void UpdateParameters(double kp, double ki, double kd);

    /**
     * @brief Compute the control signal from the current error
     * @param error Current error (setpoint - measured)
     * @return double Control signal, clamped to [-10, 10]
     *
     * Time Complexity: O(1)
     */
    double Compute(double error);

    /** @brief Reset internal state */
    void Reset();

    /**
     * @brief Get current PID parameters
     * @return Vector of [kp, ki, kd]
     */
    std::vector<double> GetParameters() const;

private:
    double kp_, ki_, kd_, dt_;
    double previousError_;
    double integral_;
    double filteredDerivative_ = 0.0;
    mutable std::mutex mutex_;
};
