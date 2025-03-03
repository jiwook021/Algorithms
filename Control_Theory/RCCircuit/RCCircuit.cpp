/**
 * @file RCCircuit.cpp
 * @brief Utility functions for RC circuit simulation
 */

#include "RCCircuit.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

void PrintDataPoint(const CircuitSimulator::DataPoint& data) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time: " << std::setw(8) << data.time << "s | "
              << "Setpoint: " << std::setw(8) << data.Setpoint << "V | "
              << "Measured: " << std::setw(8) << data.Measured << "V | "
              << "Control: " << std::setw(8) << data.Control << "V | "
              << "P: " << std::setw(8) << data.PTerm << " | "
              << "I: " << std::setw(8) << data.ITerm << " | "
              << "D: " << std::setw(8) << data.DTerm << " | "
              << "Saturated: " << (data.Saturated ? "Yes" : "No") << std::endl;
}

void SaveResultsToCSV(const std::vector<CircuitSimulator::DataPoint>& Results, const std::string& Filename) {
    std::ofstream File(Filename);
    if (!File) {
        std::cerr << "Error: Could not open file " << Filename << " for writing." << std::endl;
        return;
    }

    File << "Time,Setpoint,Measured,Control,P_Term,I_Term,D_Term,Saturated" << std::endl;

    for (const auto& data : Results) {
        File << data.time << ","
             << data.Setpoint << ","
             << data.Measured << ","
             << data.Control << ","
             << data.PTerm << ","
             << data.ITerm << ","
             << data.DTerm << ","
             << (data.Saturated ? 1 : 0) << std::endl;
    }

    std::cout << "Results saved to " << Filename << std::endl;
}

/**
 * @brief Analyze and print PID performance metrics from simulation results
 *
 * Computes six standard control-system metrics that characterize how well
 * the PID controller tracked the setpoint:
 *
 *  - **Steady-state error**: |final_value - setpoint|.
 *        How far the system ends up from the target after settling.
 *        Ideally zero; a non-zero value indicates the integral gain is
 *        too low or the system has a persistent disturbance.
 *
 *  - **Rise time (10%-90%)**: Time for the output to go from 10% to 90%
 *        of the full step change (setpoint - initial_value).
 *        Measures responsiveness.  Smaller = faster response, but
 *        pushing it too low typically increases overshoot.
 *
 *  - **Settling time (5%)**: Earliest time at which the output enters and
 *        stays within +/-5% of the setpoint for the remainder of the run.
 *        Captures how quickly oscillations or transients die out.
 *
 *  - **Maximum error**: Largest |measured - setpoint| seen during the
 *        entire run.  Dominated by the initial transient or by any
 *        disturbance injection.
 *
 *  - **RMS error**: Root-mean-square of (measured - setpoint) over all
 *        data points.  A single aggregate measure of tracking quality.
 *
 *  - **Overshoot (%)**: (peak_value - setpoint) / setpoint * 100.
 *        How far the output exceeds the target before settling.
 *        High overshoot signals too much proportional or too little
 *        derivative gain.
 *
 * @param Results  Vector of DataPoint from RunSimulation()
 * @param Setpoint The target voltage that was commanded
 */
void AnalyzePIDPerformance(const std::vector<CircuitSimulator::DataPoint>& Results, double Setpoint) {
    if (Results.empty()) {
        std::cerr << "Error: No simulation results to analyze." << std::endl;
        return;
    }

    double FinalValue = Results.back().Measured;
    double InitialValue = Results.front().Measured;
    constexpr double kSettlingThreshold = 0.05;  // 5% band around setpoint

    double MaxError = 0.0;
    double SumSquaredError = 0.0;
    double SteadyStateError = std::abs(FinalValue - Setpoint);

    // Settling time: first point where the output stays within the 5%
    // band for all remaining samples (no re-excursion).
    double SettlingTime = Results.back().time;
    bool Settled = false;
    for (size_t i = 0; i < Results.size(); ++i) {
        double Error = std::abs(Results[i].Measured - Setpoint);
        double RelativeError = Error / Setpoint;

        if (RelativeError <= kSettlingThreshold) {
            bool StaysSettled = true;
            for (size_t j = i; j < Results.size(); ++j) {
                double LaterError = std::abs(Results[j].Measured - Setpoint) / Setpoint;
                if (LaterError > kSettlingThreshold) {
                    StaysSettled = false;
                    break;
                }
            }

            if (StaysSettled) {
                Settled = true;
                SettlingTime = Results[i].time;
                break;
            }
        }
    }

    if (!Settled) {
        SettlingTime = -1.0;
    }

    // Rise time: time from 10% to 90% of the full step change
    double TargetChange = Setpoint - InitialValue;
    double RiseTime10 = -1.0;
    double RiseTime90 = -1.0;
    bool Found10 = false;
    bool Found90 = false;

    for (size_t i = 0; i < Results.size(); ++i) {
        double PercentComplete = (Results[i].Measured - InitialValue) / TargetChange;

        if (!Found10 && PercentComplete >= 0.1) {
            RiseTime10 = Results[i].time;
            Found10 = true;
        }

        if (!Found90 && PercentComplete >= 0.9) {
            RiseTime90 = Results[i].time;
            Found90 = true;
            break;
        }

        double Error = std::abs(Results[i].Measured - Setpoint);
        MaxError = std::max(MaxError, Error);
        SumSquaredError += Error * Error;
    }

    double RmsError = std::sqrt(SumSquaredError / Results.size());

    // Overshoot: how far peak output exceeds the setpoint, as a percentage
    double MaxValue = InitialValue;
    for (const auto& Point : Results) {
        MaxValue = std::max(MaxValue, Point.Measured);
    }
    double Overshoot = (MaxValue > Setpoint) ? ((MaxValue - Setpoint) / Setpoint * 100.0) : 0.0;

    std::cout << "\n=== PID Performance Analysis ===\n"
              << "Initial value      : " << InitialValue << " V\n"
              << "Final value        : " << FinalValue << " V\n"
              << "Steady-state error : " << SteadyStateError << " V ("
              << (SteadyStateError / Setpoint * 100.0) << "%)"
              << "  (how far from target after settling)\n";

    if (Found10 && Found90)
        std::cout << "Rise time (10-90%) : " << (RiseTime90 - RiseTime10) << " s"
                  << "  (speed of initial response)\n";
    else
        std::cout << "Rise time          : N/A\n";

    if (Settled)
        std::cout << "Settling time (5%) : " << SettlingTime << " s"
                  << "  (time to stay within 5% of target)\n";
    else
        std::cout << "Settling time      : N/A\n";

    std::cout << "Maximum error      : " << MaxError << " V"
              << "  (worst deviation from target)\n"
              << "RMS error          : " << RmsError << " V"
              << "  (overall tracking quality)\n"
              << "Overshoot          : " << Overshoot << "%"
              << "  (peak excursion beyond target)\n";
}
