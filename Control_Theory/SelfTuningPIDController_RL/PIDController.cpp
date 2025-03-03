/**
 * @file PIDController.cpp
 * @brief Implementation of the PIDController class
 *
 * Discrete-time PID with two safeguards:
 *   1. Integral clamping – limits the integral sum so ki * integral_sum
 *      never exceeds the output range, preventing integral windup.
 *   2. Output clamping  – hard-limits the control signal to [-10, 10],
 *      protecting the actuator from dangerously large commands.
 *
 * A low-pass filter on the derivative term smooths out noise spikes
 * that would otherwise cause aggressive control jitter.
 */

#include "PIDController.hpp"

PIDController::PIDController(double kp, double ki, double kd, double dt)
    : kp_(kp), ki_(ki), kd_(kd), dt_(dt), previousError_(0.0), integral_(0.0) {}

void PIDController::UpdateParameters(double kp, double ki, double kd) {
    std::lock_guard<std::mutex> lock(mutex_);
    kp_ = kp;
    ki_ = ki;
    kd_ = kd;
}

double PIDController::Compute(double error) {
    std::lock_guard<std::mutex> lock(mutex_);

    // --- P term: proportional to current error ---
    double pTerm = kp_ * error;

    // --- I term: accumulated error over time ---
    // Clamp the integral sum so that ki * integral never exceeds the
    // output range on its own, preventing windup when the actuator
    // is saturated and the error can't be reduced.
    integral_ += error * dt_;
    const double MAX_INTEGRAL = 10.0 / (ki_ > 0.01 ? ki_ : 0.01);
    integral_ = std::clamp(integral_, -MAX_INTEGRAL, MAX_INTEGRAL);
    double iTerm = ki_ * integral_;

    // --- D term: rate of change of error ---
    // Raw derivative = (error - prev) / dt.  A first-order low-pass
    // filter (alpha = 0.1) smooths out high-frequency noise that
    // would otherwise cause control signal jitter.
    double derivative = (error - previousError_) / dt_;
    constexpr double FILTER_COEFF = 0.1;
    filteredDerivative_ = FILTER_COEFF * derivative + (1.0 - FILTER_COEFF) * filteredDerivative_;
    double dTerm = kd_ * filteredDerivative_;

    previousError_ = error;

    // Sum P + I + D, then clamp to actuator limits
    constexpr double MAX_CONTROL = 10.0;
    return std::clamp(pTerm + iTerm + dTerm, -MAX_CONTROL, MAX_CONTROL);
}

void PIDController::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    previousError_ = 0.0;
    integral_ = 0.0;
    filteredDerivative_ = 0.0;
}

std::vector<double> PIDController::GetParameters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {kp_, ki_, kd_};
}
