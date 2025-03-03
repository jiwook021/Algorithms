/**
 * @file PIDController.cpp
 * @brief Implementation of PIDController
 *
 * Discrete-time PID with three safeguards:
 *   1. Output clamping  – keeps the control signal within actuator limits.
 *   2. Integral windup guard – freezes integral accumulation when the output
 *      is saturated, preventing the integral term from growing unbounded.
 *   3. Derivative rate limit – caps the derivative contribution to avoid
 *      spikes caused by noisy or discontinuous measurements.
 */

#include "PIDController.hpp"

PIDController::PIDController(double kp, double ki, double kd, double output_min, double output_max)
    : kp_(kp), ki_(ki), kd_(kd), output_min_(output_min), output_max_(output_max),
      last_error_(0.0), integral_sum_(0.0),
      proportional_term_(0.0), integral_term_(0.0), derivative_term_(0.0),
      first_compute_(true), is_saturated_(false) {
    if (kp < 0.0 || ki < 0.0 || kd < 0.0) {
        throw std::invalid_argument("PID gains must be non-negative");
    }
    if (output_min >= output_max) {
        throw std::invalid_argument("Output minimum must be less than maximum");
    }
}

double PIDController::Compute(double setpoint, double measured, std::optional<double> dt) {
    std::lock_guard<std::mutex> lock(mutex_);

    // error = how far we are from the target
    double error = setpoint - measured;

    // Determine the time step: use the caller-supplied dt, or fall back to
    // wall-clock elapsed time between successive Compute() calls.
    double time_delta;
    if (dt.has_value()) {
        time_delta = dt.value();
        if (time_delta <= 0.0) {
            throw std::invalid_argument("Time delta must be positive");
        }
    } else {
        auto now = std::chrono::high_resolution_clock::now();
        if (first_compute_) {
            time_delta = 0.0;
            first_compute_ = false;
        } else {
            time_delta = std::chrono::duration<double>(now - last_time_).count();
        }
        last_time_ = now;
    }

    // Guard against zero-length time steps (division-by-zero in derivative)
    if (time_delta < 1e-10) {
        time_delta = 1e-10;
    }

    // --- P term: proportional to current error ---
    proportional_term_ = kp_ * error;

    // --- I term: accumulated error over time ---
    // Only accumulate when the output is NOT saturated; otherwise the
    // integral would keep growing while the actuator can't respond,
    // causing massive overshoot once the error reverses ("windup").
    if (time_delta > 0.0 && !is_saturated_) {
        // Limit the integral sum to 1.5x the range that ki alone could fill,
        // providing a secondary safety net against runaway accumulation.
        double max_integral_sum = (output_max_ - output_min_) / (ki_ > 1e-10 ? ki_ : 1e-10) * 1.5;
        integral_sum_ += error * time_delta;
        integral_sum_ = std::clamp(integral_sum_, -max_integral_sum, max_integral_sum);
    }
    integral_term_ = ki_ * integral_sum_;

    // --- D term: rate of change of error ---
    // Clamped to ±1000 to suppress spikes from noisy measurements or
    // sudden setpoint changes.
    if (time_delta > 0.0) {
        double raw_derivative = (error - last_error_) / time_delta;
        static constexpr double kMaxDerivative = 1000.0;
        raw_derivative = std::clamp(raw_derivative, -kMaxDerivative, kMaxDerivative);
        derivative_term_ = kd_ * raw_derivative;
    } else {
        derivative_term_ = 0.0;
    }

    last_error_ = error;

    // Sum P + I + D, then clamp to actuator limits
    double output = proportional_term_ + integral_term_ + derivative_term_;
    double limited_output = std::clamp(output, output_min_, output_max_);

    // Saturation detection: output is clamped AND the error is still pushing
    // in the same direction → the actuator cannot follow the controller.
    is_saturated_ = (limited_output >= output_max_ && error > 0) ||
                    (limited_output <= output_min_ && error < 0);

    return limited_output;
}

void PIDController::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    last_error_ = 0.0;
    integral_sum_ = 0.0;
    proportional_term_ = 0.0;
    integral_term_ = 0.0;
    derivative_term_ = 0.0;
    first_compute_ = true;
    is_saturated_ = false;
}

void PIDController::SetGains(double kp, double ki, double kd) {
    if (kp < 0.0 || ki < 0.0 || kd < 0.0) {
        throw std::invalid_argument("PID gains must be non-negative");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    kp_ = kp;
    ki_ = ki;
    kd_ = kd;
}

void PIDController::SetOutputLimits(double min, double max) {
    if (min >= max) {
        throw std::invalid_argument("Output minimum must be less than maximum");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    output_min_ = min;
    output_max_ = max;
}

double PIDController::GetLastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_error_;
}

double PIDController::GetProportionalTerm() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return proportional_term_;
}

double PIDController::GetIntegralTerm() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return integral_term_;
}

double PIDController::GetDerivativeTerm() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return derivative_term_;
}

bool PIDController::IsSaturated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_saturated_;
}
