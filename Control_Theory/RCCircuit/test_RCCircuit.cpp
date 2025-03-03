/**
 * @file test_RCCircuit.cpp
 * @brief Unit tests for the RCCircuit module
 * @details Tests the PIDController, ElectricCircuit, and CircuitSimulator APIs
 *          with defined inputs and expected outputs.
 */

#include "RCCircuit.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

/// Tolerance for floating-point comparisons
static constexpr double TOLERANCE = 1e-6;
/// Relaxed tolerance for simulation convergence checks
static constexpr double SIM_TOLERANCE = 0.5;

static int TestsPassed = 0;
static int TestsFailed = 0;

/**
 * @brief Assert that two doubles are approximately equal
 * @param actual The actual value
 * @param expected The expected value
 * @param tol Tolerance
 * @param testName Name of the test
 */
static void AssertNear(double actual, double expected, double tol, const std::string& testName) {
    if (std::abs(actual - expected) <= tol) {
        std::cout << "[PASS] " << testName << std::endl;
        ++TestsPassed;
    } else {
        std::cerr << "[FAIL] " << testName
                  << " | expected: " << expected
                  << " | actual: " << actual
                  << " | tol: " << tol << std::endl;
        ++TestsFailed;
    }
}

/**
 * @brief Assert a boolean condition
 * @param condition The condition to check
 * @param testName Name of the test
 */
static void AssertTrue(bool condition, const std::string& testName) {
    if (condition) {
        std::cout << "[PASS] " << testName << std::endl;
        ++TestsPassed;
    } else {
        std::cerr << "[FAIL] " << testName << std::endl;
        ++TestsFailed;
    }
}

/**
 * @brief Assert that a callable throws the expected exception type
 * @param callable The function to invoke
 * @param testName Name of the test
 */
template <typename ExceptionType, typename Callable>
static void AssertThrows(Callable callable, const std::string& testName) {
    try {
        callable();
        std::cerr << "[FAIL] " << testName << " (no exception thrown)" << std::endl;
        ++TestsFailed;
    } catch (const ExceptionType&) {
        std::cout << "[PASS] " << testName << std::endl;
        ++TestsPassed;
    } catch (...) {
        std::cerr << "[FAIL] " << testName << " (wrong exception type)" << std::endl;
        ++TestsFailed;
    }
}

// ---------------------------------------------------------------------------
// PIDController Tests
// ---------------------------------------------------------------------------

void TestPidProportionalOnly() {
    PIDController pid(2.0, 0.0, 0.0, 0.0, 100.0);
    double output = pid.Compute(10.0, 0.0, 0.01);
    // P-only: output = Kp * error = 2.0 * 10.0 = 20.0
    AssertNear(output, 20.0, TOLERANCE, "PID: Proportional-only output");
}

void TestPidIntegralAccumulation() {
    PIDController pid(0.0, 1.0, 0.0, -1000.0, 1000.0);
    pid.Compute(10.0, 0.0, 1.0); // integral = 10.0
    double output = pid.Compute(10.0, 0.0, 1.0); // integral = 20.0
    // I-only: output = Ki * integral = 1.0 * 20.0 = 20.0
    AssertNear(output, 20.0, TOLERANCE, "PID: Integral accumulates over time");
}

void TestPidDerivativeReaction() {
    PIDController pid(0.0, 0.0, 1.0, -1000.0, 1000.0);
    pid.Compute(0.0, 0.0, 1.0); // error=0, lastError=0
    double output = pid.Compute(10.0, 0.0, 1.0); // error=10, derivative = (10-0)/1 = 10
    // D-only: output = Kd * (error - lastError) / dt = 1.0 * 10.0 = 10.0
    AssertNear(output, 10.0, TOLERANCE, "PID: Derivative reacts to error change");
}

void TestPidOutputClamping() {
    PIDController pid(100.0, 0.0, 0.0, -5.0, 5.0);
    double output = pid.Compute(10.0, 0.0, 0.01);
    // 100 * 10 = 1000, clamped to 5.0
    AssertNear(output, 5.0, TOLERANCE, "PID: Output clamped to max");
    AssertTrue(pid.IsSaturated(), "PID: Saturation detected when output clamped");
}

void TestPidOutputClampingMin() {
    PIDController pid(100.0, 0.0, 0.0, -5.0, 5.0);
    double output = pid.Compute(-10.0, 0.0, 0.01);
    AssertNear(output, -5.0, TOLERANCE, "PID: Output clamped to min");
}

void TestPidReset() {
    PIDController pid(1.0, 1.0, 1.0, -100.0, 100.0);
    pid.Compute(10.0, 0.0, 1.0);
    pid.Reset();
    AssertNear(pid.GetIntegralTerm(), 0.0, TOLERANCE, "PID: Integral cleared after reset");
    AssertNear(pid.GetProportionalTerm(), 0.0, TOLERANCE, "PID: P-term cleared after reset");
    AssertNear(pid.GetDerivativeTerm(), 0.0, TOLERANCE, "PID: D-term cleared after reset");
}

void TestPidSetGains() {
    PIDController pid(1.0, 0.0, 0.0, 0.0, 100.0);
    pid.SetGains(3.0, 0.0, 0.0);
    double output = pid.Compute(5.0, 0.0, 0.01);
    AssertNear(output, 15.0, TOLERANCE, "PID: SetGains updates proportional gain");
}

void TestPidSetOutputLimits() {
    PIDController pid(10.0, 0.0, 0.0, -100.0, 100.0);
    pid.SetOutputLimits(-2.0, 2.0);
    double output = pid.Compute(5.0, 0.0, 0.01);
    AssertNear(output, 2.0, TOLERANCE, "PID: SetOutputLimits changes clamp range");
}

void TestPidInvalidGainsThrow() {
    AssertThrows<std::invalid_argument>([] {
        PIDController pid(-1.0, 0.0, 0.0, 0.0, 10.0);
    }, "PID: Negative Kp throws invalid_argument");
}

void TestPidInvalidLimitsThrow() {
    AssertThrows<std::invalid_argument>([] {
        PIDController pid(1.0, 0.0, 0.0, 10.0, 5.0);
    }, "PID: OutputMin >= OutputMax throws invalid_argument");
}

void TestPidZeroDtThrows() {
    PIDController pid(1.0, 0.0, 0.0, 0.0, 10.0);
    AssertThrows<std::invalid_argument>([&] {
        pid.Compute(5.0, 0.0, 0.0);
    }, "PID: Zero dt throws invalid_argument");
}

// ---------------------------------------------------------------------------
// ElectricCircuit Tests
// ---------------------------------------------------------------------------

void TestCircuitInitialVoltage() {
    ElectricCircuit circuit(3.0, 10.0, 0.05, 0.0);
    AssertNear(circuit.GetVoltage(), 3.0, TOLERANCE, "Circuit: Initial voltage is set correctly");
}

void TestCircuitInitialCurrent() {
    ElectricCircuit circuit(5.0, 10.0, 0.05, 0.0);
    // I = V / R = 5 / 10 = 0.5
    AssertNear(circuit.GetCurrent(), 0.5, TOLERANCE, "Circuit: Initial current = V/R");
}

void TestCircuitApplyVoltageConverges() {
    // Zero noise so behavior is deterministic
    ElectricCircuit circuit(0.0, 10.0, 0.05, 0.0);
    // Apply a constant 5V for many steps; should converge toward 5V
    for (int i = 0; i < 5000; ++i) {
        circuit.ApplyVoltage(5.0, 0.01);
    }
    AssertNear(circuit.GetVoltage(), 5.0, 0.1, "Circuit: Voltage converges to applied voltage");
}

void TestCircuitDisturbance() {
    ElectricCircuit circuit(5.0, 10.0, 0.05, 0.0);
    circuit.AddDisturbance(-2.0);
    AssertNear(circuit.GetVoltage(), 3.0, TOLERANCE, "Circuit: Disturbance shifts voltage");
}

void TestCircuitInvalidResistanceThrows() {
    AssertThrows<std::invalid_argument>([] {
        ElectricCircuit circuit(0.0, 0.0, 0.05, 0.0);
    }, "Circuit: Zero resistance throws");
}

void TestCircuitInvalidCapacitanceThrows() {
    AssertThrows<std::invalid_argument>([] {
        ElectricCircuit circuit(0.0, 10.0, -1.0, 0.0);
    }, "Circuit: Negative capacitance throws");
}

void TestCircuitInvalidNoiseLevelThrows() {
    AssertThrows<std::invalid_argument>([] {
        ElectricCircuit circuit(0.0, 10.0, 0.05, 1.5);
    }, "Circuit: Noise level > 1.0 throws");
}

void TestCircuitInvalidDtThrows() {
    ElectricCircuit circuit(0.0, 10.0, 0.05, 0.0);
    AssertThrows<std::invalid_argument>([&] {
        circuit.ApplyVoltage(5.0, -0.01);
    }, "Circuit: Negative dt throws");
}

// ---------------------------------------------------------------------------
// CircuitSimulator Tests
// ---------------------------------------------------------------------------

void TestSimulatorStepResponse() {
    auto pid = std::make_shared<PIDController>(2.0, 0.5, 0.1, 0.0, 12.0);
    auto circuit = std::make_shared<ElectricCircuit>(0.0, 10.0, 0.05, 0.0);
    CircuitSimulator sim(pid, circuit);

    auto results = sim.RunSimulation(5.0, 10.0, 0.05);
    AssertTrue(!results.empty(), "Simulator: RunSimulation returns non-empty results");

    // After enough time, measured should approach setpoint
    double finalMeasured = results.back().Measured;
    AssertNear(finalMeasured, 5.0, SIM_TOLERANCE, "Simulator: Final voltage near setpoint");
}

void TestSimulatorDataPointFields() {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    auto circuit = std::make_shared<ElectricCircuit>(0.0, 10.0, 0.05, 0.0);
    CircuitSimulator sim(pid, circuit);

    auto results = sim.RunSimulation(5.0, 1.0, 0.1);
    AssertTrue(results.size() > 0, "Simulator: At least one data point");
    AssertNear(results.front().time, 0.0, TOLERANCE, "Simulator: First data point time is 0");
    AssertNear(results.front().Setpoint, 5.0, TOLERANCE, "Simulator: Setpoint stored correctly");
}

void TestSimulatorNullPidThrows() {
    auto circuit = std::make_shared<ElectricCircuit>(0.0, 10.0, 0.05, 0.0);
    AssertThrows<std::invalid_argument>([&] {
        CircuitSimulator sim(nullptr, circuit);
    }, "Simulator: Null PID pointer throws");
}

void TestSimulatorNullCircuitThrows() {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    AssertThrows<std::invalid_argument>([&] {
        CircuitSimulator sim(pid, nullptr);
    }, "Simulator: Null circuit pointer throws");
}

void TestSimulatorInvalidDurationThrows() {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    auto circuit = std::make_shared<ElectricCircuit>(0.0, 10.0, 0.05, 0.0);
    CircuitSimulator sim(pid, circuit);
    AssertThrows<std::invalid_argument>([&] {
        sim.RunSimulation(5.0, -1.0, 0.1);
    }, "Simulator: Negative duration throws");
}

void TestSimulatorInvalidTimeStepThrows() {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    auto circuit = std::make_shared<ElectricCircuit>(0.0, 10.0, 0.05, 0.0);
    CircuitSimulator sim(pid, circuit);
    AssertThrows<std::invalid_argument>([&] {
        sim.RunSimulation(5.0, 1.0, 0.0);
    }, "Simulator: Zero time step throws");
}

void TestSimulatorCallbackInvoked() {
    auto pid = std::make_shared<PIDController>(1.0, 0.0, 0.0, 0.0, 10.0);
    auto circuit = std::make_shared<ElectricCircuit>(0.0, 10.0, 0.05, 0.0);
    CircuitSimulator sim(pid, circuit);

    int callbackCount = 0;
    auto results = sim.RunSimulation(5.0, 1.0, 0.1,
        [&callbackCount](const CircuitSimulator::DataPoint&) { ++callbackCount; });

    AssertTrue(callbackCount == static_cast<int>(results.size()),
              "Simulator: Callback invoked for each data point");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== RCCircuit Unit Tests ===" << std::endl << std::endl;

    // PIDController tests
    TestPidProportionalOnly();
    TestPidIntegralAccumulation();
    TestPidDerivativeReaction();
    TestPidOutputClamping();
    TestPidOutputClampingMin();
    TestPidReset();
    TestPidSetGains();
    TestPidSetOutputLimits();
    TestPidInvalidGainsThrow();
    TestPidInvalidLimitsThrow();
    TestPidZeroDtThrows();

    // ElectricCircuit tests
    TestCircuitInitialVoltage();
    TestCircuitInitialCurrent();
    TestCircuitApplyVoltageConverges();
    TestCircuitDisturbance();
    TestCircuitInvalidResistanceThrows();
    TestCircuitInvalidCapacitanceThrows();
    TestCircuitInvalidNoiseLevelThrows();
    TestCircuitInvalidDtThrows();

    // CircuitSimulator tests
    TestSimulatorStepResponse();
    TestSimulatorDataPointFields();
    TestSimulatorNullPidThrows();
    TestSimulatorNullCircuitThrows();
    TestSimulatorInvalidDurationThrows();
    TestSimulatorInvalidTimeStepThrows();
    TestSimulatorCallbackInvoked();

    std::cout << std::endl << "=== Results: " << TestsPassed << " passed, "
              << TestsFailed << " failed ===" << std::endl;

    return TestsFailed > 0 ? 1 : 0;
}
