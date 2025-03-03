/**
 * @file main.cpp
 * @brief Interactive driver for PID-controlled RLC circuit simulation
 * @details Prompts the user for R, L, C component values, then runs
 *          step-response simulation with PID performance analysis.
 */

#include "PIDController.hpp"
#include "CircuitModel.hpp"
#include "CircuitSimulator.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

/**
 * @brief Read a positive double from stdin, using a default if the user presses Enter.
 * @param prompt Label shown to the user (e.g. "R (ohms)")
 * @param default_val Value used when the user presses Enter without typing
 * @return The validated positive value
 */
static double ReadPositiveValue(const std::string& prompt, double default_val) {
    while (true) {
        std::cout << "  " << prompt << " [default " << default_val << "]: ";
        std::string line;
        std::getline(std::cin, line);

        // Empty input → use default
        if (line.empty()) {
            std::cout << "    -> using " << default_val << "\n";
            return default_val;
        }

        try {
            double val = std::stod(line);
            if (val <= 0.0) {
                std::cout << "    Value must be positive. Try again.\n";
                continue;
            }
            return val;
        } catch (...) {
            std::cout << "    Invalid number. Try again.\n";
        }
    }
}

int main() {
    try {
        std::cout << "=============================================\n"
                  << "  PID-Controlled RLC Circuit Simulation\n"
                  << "=============================================\n\n"

                  << "PURPOSE\n"
                  << "  This program simulates a series RLC circuit (resistor, inductor,\n"
                  << "  capacitor) controlled by a PID feedback loop.  A PID controller\n"
                  << "  continuously adjusts the input voltage so the capacitor voltage\n"
                  << "  tracks a target setpoint, demonstrating closed-loop control of a\n"
                  << "  second-order dynamic system.\n\n"

                  << "HOW IT WORKS\n"
                  << "  The RLC circuit is governed by two coupled ODEs:\n"
                  << "    di_L/dt = (V_in - R*i_L - V_C) / L   (inductor current)\n"
                  << "    dV_C/dt = i_L / C                     (capacitor voltage)\n"
                  << "  At each time step the PID controller reads V_C, computes a\n"
                  << "  control voltage V_in, and feeds it back into the circuit.\n"
                  << "  The simulator uses sub-stepped Euler integration (10 sub-steps\n"
                  << "  per main step) for numerical stability of these stiff ODEs.\n\n"

                  << "KEY METRICS REPORTED\n"
                  << "  Resonant frequency  f0 = 1/(2*pi*sqrt(L*C))  -- natural oscillation\n"
                  << "  Damping ratio     zeta = R/(2*sqrt(L/C))     -- oscillation decay\n"
                  << "    zeta < 1 : underdamped  (oscillates before settling)\n"
                  << "    zeta = 1 : critically damped (fastest non-oscillatory settling)\n"
                  << "    zeta > 1 : overdamped  (sluggish, no oscillation)\n\n"

                  << "RECOMMENDED VALUES TO TRY\n"
                  << "  Underdamped (oscillations):  R=2   L=0.5  C=0.05  -> zeta~0.63\n"
                  << "  Critically damped:           R=4   L=0.2  C=0.05  -> zeta~1.0\n"
                  << "  Overdamped (sluggish):        R=20  L=0.2  C=0.05  -> zeta~5.0\n"
                  << "  High-Q resonance:             R=1   L=1.0  C=0.01  -> zeta~0.05\n"
                  << "---------------------------------------------\n\n";

        // --- Prompt user for RLC component values ---
        std::cout << "Enter RLC component values (press Enter for defaults):\n\n";

        double R = ReadPositiveValue("R  - Resistance  (ohms)  ", 10.0);
        double L = ReadPositiveValue("L  - Inductance  (henries)", 0.2);
        double C = ReadPositiveValue("C  - Capacitance (farads) ", 0.05);

        std::cout << "\nEnter simulation parameters:\n\n";

        double setpoint = ReadPositiveValue("Target voltage (V)", 5.0);
        double duration = ReadPositiveValue("Duration       (s)", 30.0);

        // --- Create the RLC circuit model ---
        auto rlcCircuit = std::make_shared<RLCCircuitModel>(R, L, C, 0.002);

        std::cout << "\n---------------------------------------------\n"
                  << "  Circuit Configuration\n"
                  << "---------------------------------------------\n"
                  << "  R = " << R << " ohms\n"
                  << "  L = " << L << " H\n"
                  << "  C = " << C << " F\n"
                  << "  Resonant frequency : " << rlcCircuit->ResonantFrequency() << " Hz\n"
                  << "  Damping ratio      : " << rlcCircuit->DampingRatio() << "\n"
                  << "  Target voltage     : " << setpoint << " V\n"
                  << "  Simulation duration: " << duration << " s\n"
                  << "---------------------------------------------\n\n";

        // Classify the damping regime
        double zeta = rlcCircuit->DampingRatio();
        if (zeta < 1.0)
            std::cout << "  System is UNDERDAMPED (zeta < 1) — expect oscillations\n";
        else if (zeta > 1.0)
            std::cout << "  System is OVERDAMPED  (zeta > 1) — slow, no oscillations\n";
        else
            std::cout << "  System is CRITICALLY DAMPED (zeta = 1) — fastest settling\n";

        // --- PID controller tuned for the circuit ---
        auto pid = std::make_shared<PIDController>(0.8, 0.15, 0.03, 0.0, 12.0);
        CircuitSimulator simulator(pid, rlcCircuit);
        double timeStep = 0.005;

        std::cout << "\nRunning simulation...\n\n";

        auto results = simulator.RunSimulation(setpoint, duration, timeStep, PrintDataPoint);

        // --- Save CSV and analyze ---
        SaveResultsToCSV(results, "rlc_circuit_response.csv");
        AnalyzePIDPerformance(results, setpoint);

        // --- Disturbance / Setpoint Tracking Test ---
        std::cout << "\n---------------------------------------------\n"
                  << "  Disturbance & Setpoint Tracking Test\n"
                  << "---------------------------------------------\n";

        simulator.StartContinuousSimulation(setpoint, timeStep, PrintDataPoint);

        std::this_thread::sleep_for(std::chrono::seconds(5));

        double disturbance = -2.0;
        std::cout << "\n  >> Injecting disturbance of " << disturbance << " V ...\n\n";
        simulator.AddDisturbance(disturbance);

        std::this_thread::sleep_for(std::chrono::seconds(5));

        double newSetpoint = setpoint + 2.0;
        std::cout << "\n  >> Changing setpoint to " << newSetpoint << " V ...\n\n";
        simulator.ChangeSetpoint(newSetpoint);

        std::this_thread::sleep_for(std::chrono::seconds(5));

        simulator.StopContinuousSimulation();

        std::cout << "\nSimulation completed successfully.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
