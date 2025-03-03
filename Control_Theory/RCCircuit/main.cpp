/**
 * @file main.cpp
 * @brief Interactive driver for PID-controlled RC circuit simulation
 * @details Prompts the user for R, C component values, then runs
 *          step-response simulation with PID performance analysis.
 */

#include "RCCircuit.hpp"
#include <iostream>
#include <stdexcept>

// Forward declarations of utility functions
void PrintDataPoint(const CircuitSimulator::DataPoint& data);
void SaveResultsToCSV(const std::vector<CircuitSimulator::DataPoint>& Results,
                      const std::string& Filename);
void AnalyzePIDPerformance(const std::vector<CircuitSimulator::DataPoint>& Results,
                           double Setpoint);

/**
 * @brief Read a positive double from stdin, using a default if the user presses Enter.
 */
static double ReadPositiveValue(const std::string& prompt, double default_val) {
    while (true) {
        std::cout << "  " << prompt << " [default " << default_val << "]: ";
        std::string line;
        std::getline(std::cin, line);

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
                  << "  PID-Controlled RC Circuit Simulation\n"
                  << "=============================================\n\n"

                  << "PURPOSE\n"
                  << "  This program simulates a first-order RC circuit (resistor +\n"
                  << "  capacitor) controlled by a PID feedback loop.  A PID controller\n"
                  << "  continuously adjusts the input voltage so the capacitor voltage\n"
                  << "  tracks a target setpoint, demonstrating closed-loop control of\n"
                  << "  a first-order dynamic system.\n\n"

                  << "HOW IT WORKS\n"
                  << "  The RC circuit follows the differential equation:\n"
                  << "    dV_C/dt = (V_in - V_C) / (R * C)\n"
                  << "  where V_C is the capacitor voltage and V_in is the PID output.\n"
                  << "  The time constant tau = R * C determines how fast the circuit\n"
                  << "  responds.  The PID controller compensates for this lag to reach\n"
                  << "  the setpoint quickly with minimal overshoot.\n\n"

                  << "KEY PARAMETER\n"
                  << "  Time constant  tau = R * C  (seconds)\n"
                  << "    Small tau : fast response, easier to control\n"
                  << "    Large tau : slow response, needs higher PID gains\n\n"

                  << "RECOMMENDED VALUES TO TRY\n"
                  << "  Fast response :  R=10   C=0.01  -> tau=0.1s\n"
                  << "  Medium        :  R=10   C=0.05  -> tau=0.5s  (default)\n"
                  << "  Slow response :  R=100  C=0.1   -> tau=10s\n"
                  << "  Very slow     :  R=1000 C=0.1   -> tau=100s\n"
                  << "---------------------------------------------\n\n";

        // --- Prompt user for RC component values ---
        std::cout << "Enter RC component values (press Enter for defaults):\n\n";

        double R = ReadPositiveValue("R  - Resistance  (ohms)  ", 10.0);
        double C = ReadPositiveValue("C  - Capacitance (farads) ", 0.05);

        std::cout << "\nEnter simulation parameters:\n\n";

        double Setpoint = ReadPositiveValue("Target voltage (V)", 5.0);
        double Duration = ReadPositiveValue("Duration       (s)", 50.0);

        double tau = R * C;

        std::cout << "\n---------------------------------------------\n"
                  << "  Circuit Configuration\n"
                  << "---------------------------------------------\n"
                  << "  R = " << R << " ohms\n"
                  << "  C = " << C << " F\n"
                  << "  Time constant tau = " << tau << " s\n"
                  << "  Target voltage   : " << Setpoint << " V\n"
                  << "  Duration         : " << Duration << " s\n"
                  << "---------------------------------------------\n\n";

        // Create PID controller with properly tuned parameters
        auto Pid = std::make_shared<PIDController>(2.4, 0.48, 0.16, 0.0, 12.0);

        // Create electric circuit with slight noise for realism
        auto Circuit = std::make_shared<ElectricCircuit>(0.0, R, C, 0.005);

        CircuitSimulator Simulator(Pid, Circuit);
        double TimeStep = 0.05;

        std::cout << "Running step response simulation...\n\n";

        auto Results = Simulator.RunSimulation(Setpoint, Duration, TimeStep, PrintDataPoint);

        SaveResultsToCSV(Results, "rc_step_response.csv");
        AnalyzePIDPerformance(Results, Setpoint);

        // --- Disturbance / Setpoint Tracking Test ---
        std::cout << "\n---------------------------------------------\n"
                  << "  Disturbance & Setpoint Tracking Test\n"
                  << "---------------------------------------------\n";

        Simulator.StartContinuousSimulation(Setpoint, TimeStep, PrintDataPoint);

        std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(Duration / 3)));

        double DisturbanceMagnitude = -2.0;
        std::cout << "\n  >> Injecting disturbance of " << DisturbanceMagnitude << " V ...\n\n";
        Simulator.AddDisturbance(DisturbanceMagnitude);

        std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(Duration / 3)));

        double NewSetpoint = Setpoint + 2.0;
        std::cout << "\n  >> Changing setpoint to " << NewSetpoint << " V ...\n\n";
        Simulator.ChangeSetpoint(NewSetpoint);

        std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(Duration / 3)));

        Simulator.StopContinuousSimulation();

        std::cout << "\nSimulation completed successfully.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
