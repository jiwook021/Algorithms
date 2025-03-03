/**
 * @file CircuitModel.cpp
 * @brief Implementation of RCCircuit and RLCCircuitModel
 */

#include "CircuitModel.hpp"

// ===========================================================================
// RCCircuit
// ===========================================================================

RCCircuit::RCCircuit(double R, double C, double noise_level, double initial_voltage)
    : resistance_(R), capacitance_(C), noise_level_(noise_level),
      initial_voltage_(initial_voltage), voltage_(initial_voltage),
      current_(initial_voltage / R),
      noise_dist_(0.0, 1.0) {
    if (R <= 0.0) throw std::invalid_argument("Resistance must be positive");
    if (C <= 0.0) throw std::invalid_argument("Capacitance must be positive");
    if (noise_level < 0.0 || noise_level > 1.0) {
        throw std::invalid_argument("Noise level must be between 0.0 and 1.0");
    }
    unsigned seed = static_cast<unsigned>(
        std::chrono::system_clock::now().time_since_epoch().count());
    rng_ = std::mt19937(seed);
}

double RCCircuit::SimulateStep(double input, double dt) {
    if (dt <= 0.0) throw std::invalid_argument("Time step must be positive");
    std::lock_guard<std::mutex> lock(mutex_);

    double voltage_change = (input - voltage_) / (resistance_ * capacitance_) * dt;
    voltage_change += GenerateNoise();

    double new_voltage = voltage_ + voltage_change;
    if (std::isnan(new_voltage) || std::isinf(new_voltage)) {
        new_voltage = 0.0;
    }
    voltage_ = new_voltage;
    current_ = voltage_ / resistance_;
    return voltage_;
}

void RCCircuit::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    voltage_ = initial_voltage_;
    current_ = initial_voltage_ / resistance_;
}

double RCCircuit::GetVoltage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return voltage_;
}

void RCCircuit::AddDisturbance(double magnitude) {
    std::lock_guard<std::mutex> lock(mutex_);
    voltage_ += magnitude;
}

double RCCircuit::GetCurrent() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_;
}

void RCCircuit::SetNoiseLevel(double level) {
    if (level < 0.0 || level > 1.0) {
        throw std::invalid_argument("Noise level must be between 0.0 and 1.0");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    noise_level_ = level;
}

double RCCircuit::GenerateNoise() {
    return noise_level_ * noise_dist_(rng_);
}

// ===========================================================================
// RLCCircuitModel
// ===========================================================================

RLCCircuitModel::RLCCircuitModel(double R, double L, double C,
                                 double noise_level,
                                 double initial_voltage,
                                 double initial_current)
    : resistance_(R), inductance_(L), capacitance_(C),
      noise_level_(noise_level),
      initial_voltage_(initial_voltage), initial_current_(initial_current),
      voltage_(initial_voltage), current_(initial_current),
      noise_dist_(0.0, 1.0) {
    if (R <= 0.0) throw std::invalid_argument("Resistance must be positive");
    if (L <= 0.0) throw std::invalid_argument("Inductance must be positive");
    if (C <= 0.0) throw std::invalid_argument("Capacitance must be positive");
    if (noise_level < 0.0 || noise_level > 1.0) {
        throw std::invalid_argument("Noise level must be between 0.0 and 1.0");
    }
    unsigned seed = static_cast<unsigned>(
        std::chrono::system_clock::now().time_since_epoch().count());
    rng_ = std::mt19937(seed);
}

double RLCCircuitModel::SimulateStep(double input, double dt) {
    if (dt <= 0.0) throw std::invalid_argument("Time step must be positive");
    std::lock_guard<std::mutex> lock(mutex_);

    double capacitor_voltage = voltage_;
    const double sub_dt = dt / kNumSubsteps;

    for (int i = 0; i < kNumSubsteps; ++i) {
        double current_derivative =
            (input - current_ * resistance_ - capacitor_voltage) / inductance_;

        static constexpr double kMaxDerivative = 1e6;
        current_derivative = std::clamp(current_derivative, -kMaxDerivative, kMaxDerivative);

        double new_current = current_ + current_derivative * sub_dt;
        if (std::isnan(new_current) || std::isinf(new_current)) {
            new_current = 0.0;
        }
        current_ = new_current;

        double voltage_change = (current_ / capacitance_) * sub_dt;
        if (i == 0) {
            voltage_change += GenerateNoise() * 0.1;
        }

        double new_voltage = capacitor_voltage + voltage_change;
        if (std::isnan(new_voltage) || std::isinf(new_voltage)) {
            new_voltage = 0.0;
        }
        voltage_ = new_voltage;
        capacitor_voltage = new_voltage;
    }
    return voltage_;
}

void RLCCircuitModel::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    voltage_ = initial_voltage_;
    current_ = initial_current_;
}

double RLCCircuitModel::GetVoltage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return voltage_;
}

void RLCCircuitModel::AddDisturbance(double magnitude) {
    std::lock_guard<std::mutex> lock(mutex_);
    voltage_ += magnitude;
}

double RLCCircuitModel::ResonantFrequency() const {
    return 1.0 / (2.0 * kPi * std::sqrt(inductance_ * capacitance_));
}

double RLCCircuitModel::DampingRatio() const {
    return resistance_ / (2.0 * std::sqrt(inductance_ / capacitance_));
}

double RLCCircuitModel::GetInductorCurrent() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_;
}

double RLCCircuitModel::GenerateNoise() {
    return noise_level_ * noise_dist_(rng_);
}
