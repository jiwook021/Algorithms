/**
 * @file Environment.cpp
 * @brief Implementation of the SimpleProcessEnvironment class
 */

#include "Environment.hpp"

SimpleProcessEnvironment::SimpleProcessEnvironment(double timeConstant, double gain,
                                                     double dt, double disturbanceStdDev)
    : timeConstant_(timeConstant), gain_(gain), dt_(dt),
      disturbanceStdDev_(disturbanceStdDev),
      processValue_(0.0), setpoint_(0.0), stepCount_(0),
      maxSteps_(1000), gen_(std::random_device{}()),
      dist_(0.0, disturbanceStdDev) {}

std::vector<double> SimpleProcessEnvironment::Reset() {
    stepCount_ = 0;
    processValue_ = 0.0;
    setpoint_ = 1.0;
    return {setpoint_ - processValue_, 0.0, 0.0};
}

std::tuple<std::vector<double>, double, bool> SimpleProcessEnvironment::Step(
    const std::vector<double>& action, double controlSignal) {
    (void)action;

    constexpr double MAX_CONTROL = 10.0;
    double safeControl = std::clamp(controlSignal, -MAX_CONTROL, MAX_CONTROL);

    double prevValue = processValue_;
    double derivative = (gain_ * safeControl - processValue_) / timeConstant_;
    constexpr double MAX_RATE = 1.0;
    derivative = std::clamp(derivative, -MAX_RATE, MAX_RATE);
    processValue_ += dt_ * derivative;

    if (disturbanceStdDev_ > 0.0) {
        constexpr double MAX_DISTURBANCE = 0.1;
        processValue_ += std::clamp(dist_(gen_), -MAX_DISTURBANCE, MAX_DISTURBANCE);
    }

    double prevError = setpoint_ - prevValue;
    double error = setpoint_ - processValue_;
    constexpr double MAX_ERROR_RATE = 5.0;
    double errorDerivative = std::clamp((error - prevError) / dt_, -MAX_ERROR_RATE, MAX_ERROR_RATE);

    std::vector<double> state = {error, errorDerivative, safeControl};

    double reward = -std::min(5.0, error * error) - 0.01 * std::min(5.0, safeControl * safeControl);
    if (std::abs(error) < 0.1) reward += 0.5;

    ++stepCount_;
    bool done = stepCount_ >= maxSteps_;

    return {state, reward, done};
}

double SimpleProcessEnvironment::GetSetpoint() const { return setpoint_; }
double SimpleProcessEnvironment::GetProcessValue() const { return processValue_; }

void SimpleProcessEnvironment::SetSetpoint(double setpoint) {
    setpoint_ = std::clamp(setpoint, -5.0, 5.0);
}

void SimpleProcessEnvironment::SetMaxSteps(int maxSteps) { maxSteps_ = maxSteps; }
