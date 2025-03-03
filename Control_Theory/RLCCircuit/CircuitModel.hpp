/**
 * @file CircuitModel.hpp
 * @brief Abstract circuit model base class, RC circuit, and RLC circuit
 * @details Defines the CircuitModel interface, a first-order RCCircuit, and a
 *          second-order RLCCircuitModel with resonance and damping analytics.
 *
 * Time Complexity: O(1) per simulation step
 * Space Complexity: O(1)
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <random>
#include <stdexcept>

/** @brief Pi constant for frequency calculations */
static constexpr double kPi = 3.14159265358979323846;

/**
 * @class CircuitModel
 * @brief Abstract interface for circuit simulation models
 */
class CircuitModel {
public:
    virtual ~CircuitModel() = default;

    /**
     * @brief Simulate one time step with the given input voltage
     * @param input Input voltage (V)
     * @param dt Time step in seconds (must be > 0)
     * @return The output voltage after the step
     */
    virtual double SimulateStep(double input, double dt) = 0;

    /** @brief Reset the circuit to its initial state */
    virtual void Reset() = 0;

    /** @return Current output voltage */
    virtual double GetVoltage() const = 0;

    /**
     * @brief Inject a voltage disturbance
     * @param magnitude Disturbance magnitude (V)
     */
    virtual void AddDisturbance(double magnitude) = 0;
};

/**
 * @class RCCircuit
 * @brief First-order RC circuit model with optional noise injection
 *
 * Simulates capacitor voltage via Euler integration of dV/dt = (V_in - V_out)/(RC).
 */
class RCCircuit : public CircuitModel {
public:
    /**
     * @brief Construct an RC circuit
     * @param R Resistance in ohms (must be > 0)
     * @param C Capacitance in farads (must be > 0)
     * @param noise_level Noise level 0.0-1.0 (default 0.0)
     * @param initial_voltage Initial capacitor voltage (V, default 0.0)
     * @throws std::invalid_argument if parameters are invalid
     */
    RCCircuit(double R, double C, double noise_level = 0.0, double initial_voltage = 0.0);

    double SimulateStep(double input, double dt) override;
    void Reset() override;
    double GetVoltage() const override;
    void AddDisturbance(double magnitude) override;

    /** @return Current through the circuit */
    double GetCurrent() const;

    /**
     * @brief Set the noise level
     * @param level Noise level 0.0-1.0
     * @throws std::invalid_argument if level is out of range
     */
    void SetNoiseLevel(double level);

    /** @return Resistance in ohms */
    double GetResistance() const { return resistance_; }
    /** @return Capacitance in farads */
    double GetCapacitance() const { return capacitance_; }
    /** @return Time constant tau = R*C */
    double GetTimeConstant() const { return resistance_ * capacitance_; }

private:
    double resistance_;
    double capacitance_;
    double noise_level_;
    double initial_voltage_;
    double voltage_;
    double current_;
    std::mt19937 rng_;
    std::normal_distribution<double> noise_dist_;
    mutable std::mutex mutex_;

    double GenerateNoise();
};

/**
 * @class RLCCircuitModel
 * @brief Second-order RLC circuit model with inductance
 *
 * Extends the circuit model with an inductor, producing second-order dynamics:
 *   di_L/dt = (V_in - R*i_L - V_C) / L
 *   dV_C/dt = i_L / C
 *
 * Uses sub-stepped Euler integration for numerical stability.
 */
class RLCCircuitModel : public CircuitModel {
public:
    /**
     * @brief Construct an RLC circuit
     * @param R Resistance in ohms (must be > 0)
     * @param L Inductance in henries (must be > 0)
     * @param C Capacitance in farads (must be > 0)
     * @param noise_level Noise level 0.0-1.0 (default 0.0)
     * @param initial_voltage Initial capacitor voltage (V, default 0.0)
     * @param initial_current Initial inductor current (A, default 0.0)
     * @throws std::invalid_argument if parameters are invalid
     */
    RLCCircuitModel(double R, double L, double C,
                    double noise_level = 0.0,
                    double initial_voltage = 0.0,
                    double initial_current = 0.0);

    double SimulateStep(double input, double dt) override;
    void Reset() override;
    double GetVoltage() const override;
    void AddDisturbance(double magnitude) override;

    /** @return Resonant frequency in Hz: f0 = 1/(2*pi*sqrt(LC)) */
    double ResonantFrequency() const;

    /** @return Damping ratio: zeta = R/(2*sqrt(L/C)) */
    double DampingRatio() const;

    /** @return Inductance in henries */
    double GetInductance() const { return inductance_; }
    /** @return Resistance in ohms */
    double GetResistance() const { return resistance_; }
    /** @return Capacitance in farads */
    double GetCapacitance() const { return capacitance_; }
    /** @return Current through the inductor */
    double GetInductorCurrent() const;

private:
    double resistance_;
    double inductance_;
    double capacitance_;
    double noise_level_;
    double initial_voltage_;
    double initial_current_;
    double voltage_;
    double current_;
    std::mt19937 rng_;
    std::normal_distribution<double> noise_dist_;
    mutable std::mutex mutex_;

    static constexpr int kNumSubsteps = 10;

    double GenerateNoise();
};
