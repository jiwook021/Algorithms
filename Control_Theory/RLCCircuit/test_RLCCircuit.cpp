/**
 * @file test_RLCCircuit.cpp
 * @brief Google Test suite for the RLCCircuit module
 * @details Tests PIDController, RCCircuit, RLCCircuitModel, and CircuitSimulator
 *          with analytical verification and simulation convergence checks.
 */

#include "PIDController.hpp"
#include "CircuitModel.hpp"
#include "CircuitSimulator.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// PrintSection helper
// ---------------------------------------------------------------------------

static void PrintSection(const std::string& title) {
    std::cout << "\n========================================\n"
              << "  " << title << "\n"
              << "========================================\n";
}

// ---------------------------------------------------------------------------
// PIDController Tests
// ---------------------------------------------------------------------------

class PIDControllerTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("PIDController Tests");
    }
};

TEST_F(PIDControllerTest, ProportionalOnlyOutput) {
    PIDController pid(2.0, 0.0, 0.0, 0.0, 100.0);
    double output = pid.Compute(10.0, 0.0, 0.01);
    EXPECT_NEAR(output, 20.0, 1e-6);
}

TEST_F(PIDControllerTest, OutputClampedToMax) {
    PIDController pid(100.0, 0.0, 0.0, -5.0, 5.0);
    double output = pid.Compute(10.0, 0.0, 0.01);
    EXPECT_NEAR(output, 5.0, 1e-6);
}

TEST_F(PIDControllerTest, SaturationDetectedWhenClamped) {
    PIDController pid(100.0, 0.0, 0.0, -5.0, 5.0);
    pid.Compute(10.0, 0.0, 0.01);
    EXPECT_TRUE(pid.IsSaturated());
}

TEST_F(PIDControllerTest, ResetClearsState) {
    PIDController pid(1.0, 1.0, 1.0, -100.0, 100.0);
    pid.Compute(10.0, 0.0, 1.0);
    pid.Reset();
    EXPECT_NEAR(pid.GetIntegralTerm(), 0.0, 1e-6);
    EXPECT_NEAR(pid.GetLastError(), 0.0, 1e-6);
    EXPECT_NEAR(pid.GetProportionalTerm(), 0.0, 1e-6);
    EXPECT_NEAR(pid.GetDerivativeTerm(), 0.0, 1e-6);
}

TEST_F(PIDControllerTest, NegativeGainThrows) {
    EXPECT_THROW(PIDController(-1.0, 0.0, 0.0, 0.0, 10.0), std::invalid_argument);
    EXPECT_THROW(PIDController(0.0, -1.0, 0.0, 0.0, 10.0), std::invalid_argument);
    EXPECT_THROW(PIDController(0.0, 0.0, -1.0, 0.0, 10.0), std::invalid_argument);
}

TEST_F(PIDControllerTest, InvalidOutputLimitsThrows) {
    EXPECT_THROW(PIDController(1.0, 0.0, 0.0, 10.0, 5.0), std::invalid_argument);
    EXPECT_THROW(PIDController(1.0, 0.0, 0.0, 5.0, 5.0), std::invalid_argument);
}

TEST_F(PIDControllerTest, NegativeDtThrows) {
    PIDController pid(1.0, 0.0, 0.0, 0.0, 10.0);
    EXPECT_THROW(pid.Compute(5.0, 0.0, -0.01), std::invalid_argument);
}

TEST_F(PIDControllerTest, SetpointTrackingConverges) {
    PIDController pid(2.0, 0.5, 0.1, 0.0, 20.0);
    double measured = 0.0;
    constexpr double kSetpoint = 10.0;
    constexpr double kDt = 0.01;

    for (int i = 0; i < 5000; ++i) {
        double control = pid.Compute(kSetpoint, measured, kDt);
        // Simple first-order plant: measured moves toward control
        measured += (control - measured) * 0.1 * kDt;
    }
    EXPECT_NEAR(measured, kSetpoint, 0.5)
        << "PID should track setpoint within tolerance";
}

// ---------------------------------------------------------------------------
// RCCircuit Tests
// ---------------------------------------------------------------------------

class RCCircuitTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("RCCircuit Tests");
    }
};

TEST_F(RCCircuitTest, InitialVoltageZero) {
    RCCircuit rc(10.0, 0.05);
    EXPECT_NEAR(rc.GetVoltage(), 0.0, 1e-6);
}

TEST_F(RCCircuitTest, TimeConstant) {
    RCCircuit rc(10.0, 0.05);
    EXPECT_NEAR(rc.GetTimeConstant(), 0.5, 1e-6);
}

TEST_F(RCCircuitTest, StepResponseTimeConstant) {
    // For an RC circuit, after one time constant tau=RC the voltage should
    // reach ~63.2% of the applied step voltage (1 - e^-1).
    constexpr double kR = 10.0;
    constexpr double kC = 0.05;
    constexpr double kTau = kR * kC;  // 0.5s
    constexpr double kStepVoltage = 5.0;
    constexpr double kDt = 0.0001;

    RCCircuit rc(kR, kC, 0.0);  // No noise for deterministic test

    int steps_per_tau = static_cast<int>(kTau / kDt);
    for (int i = 0; i < steps_per_tau; ++i) {
        rc.SimulateStep(kStepVoltage, kDt);
    }

    double expected = kStepVoltage * (1.0 - std::exp(-1.0));  // ~3.16
    EXPECT_NEAR(rc.GetVoltage(), expected, 0.15)
        << "After one time constant, voltage should be ~63.2% of step";
}

TEST_F(RCCircuitTest, ConvergesToInputVoltage) {
    RCCircuit rc(10.0, 0.05, 0.0);
    for (int i = 0; i < 10000; ++i) {
        rc.SimulateStep(5.0, 0.001);
    }
    EXPECT_NEAR(rc.GetVoltage(), 5.0, 0.1);
}

TEST_F(RCCircuitTest, InvalidResistanceThrows) {
    EXPECT_THROW(RCCircuit(0.0, 0.05), std::invalid_argument);
    EXPECT_THROW(RCCircuit(-1.0, 0.05), std::invalid_argument);
}

TEST_F(RCCircuitTest, InvalidCapacitanceThrows) {
    EXPECT_THROW(RCCircuit(10.0, 0.0), std::invalid_argument);
    EXPECT_THROW(RCCircuit(10.0, -0.01), std::invalid_argument);
}

TEST_F(RCCircuitTest, NegativeDtThrows) {
    RCCircuit rc(10.0, 0.05);
    EXPECT_THROW(rc.SimulateStep(5.0, -0.01), std::invalid_argument);
}

TEST_F(RCCircuitTest, Disturbance) {
    RCCircuit rc(10.0, 0.05, 0.0, 5.0);
    rc.AddDisturbance(-2.0);
    EXPECT_NEAR(rc.GetVoltage(), 3.0, 1e-6);
}

TEST_F(RCCircuitTest, ResetRestoresInitialState) {
    RCCircuit rc(10.0, 0.05, 0.0, 0.0);
    for (int i = 0; i < 1000; ++i) {
        rc.SimulateStep(5.0, 0.001);
    }
    rc.Reset();
    EXPECT_NEAR(rc.GetVoltage(), 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// RLCCircuitModel Tests
// ---------------------------------------------------------------------------

class RLCCircuitModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("RLCCircuitModel Tests");
    }
};

TEST_F(RLCCircuitModelTest, InitialState) {
    RLCCircuitModel rlc(100.0, 0.1, 0.001);
    EXPECT_NEAR(rlc.GetVoltage(), 0.0, 1e-6);
    EXPECT_NEAR(rlc.GetInductorCurrent(), 0.0, 1e-6);
}

TEST_F(RLCCircuitModelTest, ResonantFrequencyFormula) {
    constexpr double kR = 100.0;
    constexpr double kL = 0.1;
    constexpr double kC = 0.001;
    RLCCircuitModel rlc(kR, kL, kC);

    double expected_freq = 1.0 / (2.0 * kPi * std::sqrt(kL * kC));
    EXPECT_NEAR(rlc.ResonantFrequency(), expected_freq, 0.01)
        << "Resonant frequency should match 1/(2*pi*sqrt(LC))";
}

TEST_F(RLCCircuitModelTest, ResonantFrequencyDifferentValues) {
    // With L=1.0 H, C=1.0 F, f0 = 1/(2*pi) ~ 0.159 Hz
    RLCCircuitModel rlc(10.0, 1.0, 1.0);
    double expected = 1.0 / (2.0 * kPi);
    EXPECT_NEAR(rlc.ResonantFrequency(), expected, 0.001);
}

TEST_F(RLCCircuitModelTest, DampingRatioFormula) {
    constexpr double kR = 100.0;
    constexpr double kL = 0.1;
    constexpr double kC = 0.001;
    RLCCircuitModel rlc(kR, kL, kC);

    double expected_zeta = kR / (2.0 * std::sqrt(kL / kC));
    EXPECT_NEAR(rlc.DampingRatio(), expected_zeta, 0.01)
        << "Damping ratio should match R/(2*sqrt(L/C))";
}

TEST_F(RLCCircuitModelTest, DampingRatioPositive) {
    RLCCircuitModel rlc(10.0, 0.5, 0.01);
    EXPECT_GT(rlc.DampingRatio(), 0.0);
}

TEST_F(RLCCircuitModelTest, UnderdampedSystem) {
    // R small relative to sqrt(L/C) => zeta < 1
    // R=1, L=1, C=0.01 => sqrt(L/C) = 10 => zeta = 1/(2*10) = 0.05
    RLCCircuitModel rlc(1.0, 1.0, 0.01);
    EXPECT_LT(rlc.DampingRatio(), 1.0)
        << "Low R should produce underdamped system (zeta < 1)";
}

TEST_F(RLCCircuitModelTest, OverdampedSystem) {
    // R large relative to sqrt(L/C) => zeta > 1
    // R=100, L=0.1, C=0.01 => sqrt(L/C) = sqrt(10) ~ 3.16 => zeta = 100/(2*3.16) ~ 15.8
    RLCCircuitModel rlc(100.0, 0.1, 0.01);
    EXPECT_GT(rlc.DampingRatio(), 1.0)
        << "High R should produce overdamped system (zeta > 1)";
}

TEST_F(RLCCircuitModelTest, CriticallyDampedSystem) {
    // zeta = 1 when R = 2*sqrt(L/C)
    // L=1, C=1 => sqrt(L/C) = 1 => R = 2
    RLCCircuitModel rlc(2.0, 1.0, 1.0);
    EXPECT_NEAR(rlc.DampingRatio(), 1.0, 0.01)
        << "R = 2*sqrt(L/C) should produce critically damped system (zeta ~ 1)";
}

TEST_F(RLCCircuitModelTest, VoltageResponseConverges) {
    // Overdamped for reliable convergence
    RLCCircuitModel rlc(10.0, 0.1, 0.01, 0.0);
    for (int i = 0; i < 2000; ++i) {
        rlc.SimulateStep(5.0, 0.001);
    }
    EXPECT_NEAR(rlc.GetVoltage(), 5.0, 0.5)
        << "RLC voltage should converge to applied voltage";
}

TEST_F(RLCCircuitModelTest, StepResponseCurrentFlows) {
    RLCCircuitModel rlc(10.0, 0.1, 0.01, 0.0);
    rlc.SimulateStep(5.0, 0.001);
    EXPECT_GT(rlc.GetInductorCurrent(), 0.0)
        << "Current should flow after voltage applied";
}

TEST_F(RLCCircuitModelTest, InvalidInductanceThrows) {
    EXPECT_THROW(RLCCircuitModel(10.0, -0.1, 0.01), std::invalid_argument);
    EXPECT_THROW(RLCCircuitModel(10.0, 0.0, 0.01), std::invalid_argument);
}

TEST_F(RLCCircuitModelTest, InvalidResistanceThrows) {
    EXPECT_THROW(RLCCircuitModel(0.0, 0.1, 0.01), std::invalid_argument);
    EXPECT_THROW(RLCCircuitModel(-5.0, 0.1, 0.01), std::invalid_argument);
}

TEST_F(RLCCircuitModelTest, InvalidCapacitanceThrows) {
    EXPECT_THROW(RLCCircuitModel(10.0, 0.1, 0.0), std::invalid_argument);
}

TEST_F(RLCCircuitModelTest, NegativeDtThrows) {
    RLCCircuitModel rlc(10.0, 0.1, 0.01);
    EXPECT_THROW(rlc.SimulateStep(5.0, -0.01), std::invalid_argument);
}

TEST_F(RLCCircuitModelTest, GetComponentValues) {
    RLCCircuitModel rlc(10.0, 0.5, 0.01);
    EXPECT_NEAR(rlc.GetResistance(), 10.0, 1e-6);
    EXPECT_NEAR(rlc.GetInductance(), 0.5, 1e-6);
    EXPECT_NEAR(rlc.GetCapacitance(), 0.01, 1e-6);
}

TEST_F(RLCCircuitModelTest, Disturbance) {
    RLCCircuitModel rlc(10.0, 0.1, 0.01, 0.0, 5.0);
    rlc.AddDisturbance(-2.0);
    EXPECT_NEAR(rlc.GetVoltage(), 3.0, 1e-6);
}

TEST_F(RLCCircuitModelTest, ResetRestoresInitialState) {
    RLCCircuitModel rlc(10.0, 0.1, 0.01, 0.0, 0.0, 0.0);
    for (int i = 0; i < 1000; ++i) {
        rlc.SimulateStep(5.0, 0.001);
    }
    rlc.Reset();
    EXPECT_NEAR(rlc.GetVoltage(), 0.0, 1e-6);
    EXPECT_NEAR(rlc.GetInductorCurrent(), 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// CircuitSimulator Tests
// ---------------------------------------------------------------------------

class CircuitSimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        PrintSection("CircuitSimulator Tests");
    }
};

TEST_F(CircuitSimulatorTest, RunReturnsCorrectVectorSizes) {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    auto rlc = std::make_shared<RLCCircuitModel>(10.0, 0.1, 0.01, 0.0);
    CircuitSimulator sim(pid, rlc);

    auto result = sim.Run(5.0, 1.0, 0.1);
    // duration=1.0, step=0.1 => 11 points (0.0, 0.1, ..., 1.0)
    EXPECT_EQ(result.time.size(), 11u);
    EXPECT_EQ(result.setpoint.size(), 11u);
    EXPECT_EQ(result.output.size(), 11u);
    EXPECT_EQ(result.control_signal.size(), 11u);
}

TEST_F(CircuitSimulatorTest, RunSimulationReturnsCorrectCount) {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    auto rlc = std::make_shared<RLCCircuitModel>(10.0, 0.1, 0.01, 0.0);
    CircuitSimulator sim(pid, rlc);

    auto results = sim.RunSimulation(5.0, 1.0, 0.1);
    EXPECT_EQ(results.size(), 11u);
}

TEST_F(CircuitSimulatorTest, SimulatorWithRLCConverges) {
    auto pid = std::make_shared<PIDController>(0.8, 0.15, 0.03, 0.0, 12.0);
    auto rlc = std::make_shared<RLCCircuitModel>(4.47, 0.2, 0.05, 0.0);
    CircuitSimulator sim(pid, rlc);

    auto result = sim.Run(5.0, 30.0, 0.01);
    ASSERT_FALSE(result.output.empty());
    double final_voltage = result.output.back();
    EXPECT_NEAR(final_voltage, 5.0, 0.5)
        << "Simulator+RLC should converge to setpoint";
}

TEST_F(CircuitSimulatorTest, SimulatorWithRCConverges) {
    auto pid = std::make_shared<PIDController>(1.0, 0.5, 0.05, 0.0, 12.0);
    auto rc = std::make_shared<RCCircuit>(10.0, 0.05, 0.0);
    CircuitSimulator sim(pid, rc);

    auto result = sim.Run(5.0, 30.0, 0.01);
    ASSERT_FALSE(result.output.empty());
    double final_voltage = result.output.back();
    EXPECT_NEAR(final_voltage, 5.0, 0.5)
        << "Simulator+RC should converge to setpoint";
}

TEST_F(CircuitSimulatorTest, NullPIDThrows) {
    auto rlc = std::make_shared<RLCCircuitModel>(10.0, 0.1, 0.01);
    EXPECT_THROW(CircuitSimulator(nullptr, rlc), std::invalid_argument);
}

TEST_F(CircuitSimulatorTest, NullCircuitThrows) {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    EXPECT_THROW(CircuitSimulator(pid, nullptr), std::invalid_argument);
}

TEST_F(CircuitSimulatorTest, InvalidDurationThrows) {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    auto rlc = std::make_shared<RLCCircuitModel>(10.0, 0.1, 0.01);
    CircuitSimulator sim(pid, rlc);
    EXPECT_THROW(sim.Run(5.0, -1.0, 0.01), std::invalid_argument);
}

TEST_F(CircuitSimulatorTest, InvalidTimeStepThrows) {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    auto rlc = std::make_shared<RLCCircuitModel>(10.0, 0.1, 0.01);
    CircuitSimulator sim(pid, rlc);
    EXPECT_THROW(sim.Run(5.0, 1.0, -0.01), std::invalid_argument);
}

TEST_F(CircuitSimulatorTest, SetpointVectorMatchesTarget) {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    auto rlc = std::make_shared<RLCCircuitModel>(10.0, 0.1, 0.01, 0.0);
    CircuitSimulator sim(pid, rlc);

    auto result = sim.Run(7.5, 1.0, 0.1);
    for (double sp : result.setpoint) {
        EXPECT_NEAR(sp, 7.5, 1e-6);
    }
}
