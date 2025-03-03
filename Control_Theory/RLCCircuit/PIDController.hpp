/**
 * @file PIDController.hpp
 * @brief Thread-safe PID controller with anti-windup protection
 * @details Implements a discrete-time Proportional-Integral-Derivative controller
 *          with output clamping and integral wind-up prevention.
 *
 * Time Complexity: O(1) per Compute call
 * Space Complexity: O(1)
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <stdexcept>

/**
 * @class PIDController
 * @brief Thread-safe PID controller with anti-windup protection
 *
 * Implements a discrete-time Proportional-Integral-Derivative controller
 * with output clamping and integral wind-up prevention.
 */
class PIDController {
public:
    /**
     * @brief Construct a PIDController with specified gains and output limits
     * @param kp Proportional gain (must be >= 0)
     * @param ki Integral gain (must be >= 0)
     * @param kd Derivative gain (must be >= 0)
     * @param output_min Minimum output value
     * @param output_max Maximum output value (must be > output_min)
     * @throws std::invalid_argument if gains are negative or limits are invalid
     */
    PIDController(double kp, double ki, double kd, double output_min, double output_max);

    /**
     * @brief Compute the control output
     * @param setpoint The desired target value
     * @param measured The current measured value
     * @param dt Time delta in seconds (optional; auto-calculated if omitted)
     * @return double The computed and clamped output value
     * @throws std::invalid_argument if dt is provided and <= 0
     */
    double Compute(double setpoint, double measured, std::optional<double> dt = std::nullopt);

    /** @brief Reset all internal state */
    void Reset();

    /**
     * @brief Update PID gains
     * @throws std::invalid_argument if any gain is negative
     */
    void SetGains(double kp, double ki, double kd);

    /**
     * @brief Update output limits
     * @throws std::invalid_argument if min >= max
     */
    void SetOutputLimits(double min, double max);

    /** @return The last computed error */
    double GetLastError() const;
    /** @return The proportional contribution */
    double GetProportionalTerm() const;
    /** @return The integral contribution */
    double GetIntegralTerm() const;
    /** @return The derivative contribution */
    double GetDerivativeTerm() const;
    /** @return True if output is saturated (clamped at limits while error persists) */
    bool IsSaturated() const;

private:
    double kp_, ki_, kd_;
    double output_min_, output_max_;
    double last_error_, integral_sum_;
    double proportional_term_, integral_term_, derivative_term_;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_time_;
    bool first_compute_;
    bool is_saturated_;
    mutable std::mutex mutex_;
};
