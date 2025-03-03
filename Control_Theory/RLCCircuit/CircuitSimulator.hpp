/**
 * @file CircuitSimulator.hpp
 * @brief Closed-loop simulator combining PIDController and CircuitModel
 * @details Supports batch simulation and continuous background-thread
 *          simulation with runtime setpoint changes and disturbance injection.
 *
 * Time Complexity: O(N) where N = duration/dt
 * Space Complexity: O(N) for result storage
 */

#pragma once

#include "PIDController.hpp"
#include "CircuitModel.hpp"

#include <functional>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

/**
 * @class CircuitSimulator
 * @brief Closed-loop simulator combining PIDController and CircuitModel
 */
class CircuitSimulator {
public:
    /** @brief Simulation result vectors for batch runs */
    struct SimulationResult {
        std::vector<double> time;
        std::vector<double> setpoint;
        std::vector<double> output;
        std::vector<double> control_signal;
    };

    /** @brief Detailed data point for per-step callbacks */
    struct DataPoint {
        double time;
        double Setpoint;
        double Measured;
        double Control;
        double PTerm;
        double ITerm;
        double DTerm;
        bool Saturated;
    };

    using DataCallback = std::function<void(const DataPoint&)>;

    /**
     * @brief Construct a CircuitSimulator
     * @param pid Shared PID controller (must not be null)
     * @param circuit Shared circuit model (must not be null)
     * @throws std::invalid_argument if either pointer is null
     */
    CircuitSimulator(std::shared_ptr<PIDController> pid,
                     std::shared_ptr<CircuitModel> circuit);

    ~CircuitSimulator();

    /**
     * @brief Run a batch simulation returning SimulationResult
     * @param setpoint Target voltage
     * @param duration Duration in seconds (must be > 0)
     * @param dt Time step in seconds (must be > 0)
     * @return SimulationResult with time, setpoint, output, control_signal vectors
     */
    SimulationResult Run(double setpoint, double duration, double dt);

    /**
     * @brief Run a batch simulation returning detailed DataPoint vector
     * @param setpoint Target voltage
     * @param duration Duration in seconds (must be > 0)
     * @param time_step Time step in seconds (must be > 0)
     * @param callback Optional per-step callback
     * @return Vector of DataPoint results
     */
    std::vector<DataPoint> RunSimulation(double setpoint, double duration,
                                         double time_step, DataCallback callback = nullptr);

    /**
     * @brief Start a continuous background simulation
     * @param setpoint Initial target value
     * @param time_step Time step (s)
     * @param callback Per-step callback
     */
    void StartContinuousSimulation(double setpoint, double time_step, DataCallback callback);

    /** @brief Stop the continuous simulation and join the thread */
    void StopContinuousSimulation();

    /** @return True if a continuous simulation is running */
    bool IsSimulationRunning() const;

    /** @brief Change the setpoint during a continuous simulation */
    void ChangeSetpoint(double new_setpoint);

    /** @brief Inject a disturbance into the circuit */
    void AddDisturbance(double magnitude);

private:
    std::shared_ptr<PIDController> pid_controller_;
    std::shared_ptr<CircuitModel> circuit_;
    bool is_running_;
    double current_setpoint_;
    std::thread simulation_thread_;
    mutable std::mutex mutex_;

    DataPoint RunSimulationStep(double setpoint, double current_time, double time_step);
    void ContinuousSimulationThread(double setpoint, double time_step, DataCallback callback);
};

/**
 * @brief Print a single DataPoint to stdout
 */
inline void PrintDataPoint(const CircuitSimulator::DataPoint& data) {
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

/**
 * @brief Save simulation results to a CSV file
 */
inline void SaveResultsToCSV(const std::vector<CircuitSimulator::DataPoint>& results,
                              const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    file << "Time,Setpoint,Measured,Control,P_Term,I_Term,D_Term,Saturated" << std::endl;
    for (const auto& d : results) {
        file << d.time << "," << d.Setpoint << "," << d.Measured << ","
             << d.Control << "," << d.PTerm << "," << d.ITerm << ","
             << d.DTerm << "," << (d.Saturated ? 1 : 0) << std::endl;
    }
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
 *        stays within ±5% of the setpoint for the remainder of the run.
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
 * @param results  Vector of DataPoint from RunSimulation()
 * @param setpoint The target voltage that was commanded
 */
inline void AnalyzePIDPerformance(const std::vector<CircuitSimulator::DataPoint>& results,
                                   double setpoint) {
    if (results.empty()) {
        std::cerr << "Error: No simulation results to analyze." << std::endl;
        return;
    }

    double final_value = results.back().Measured;
    double initial_value = results.front().Measured;
    constexpr double kSettlingThreshold = 0.05;  // 5% band around setpoint

    double max_error = 0.0;
    double sum_squared_error = 0.0;
    double steady_state_error = std::abs(final_value - setpoint);
    double settling_time = -1.0;

    for (size_t i = 0; i < results.size(); ++i) {
        double error = std::abs(results[i].Measured - setpoint);
        double relative_error = (std::abs(setpoint) > 1e-6) ? error / std::abs(setpoint) : error;

        // Settling time: first point where the output stays within the 5%
        // band for all remaining samples (no re-excursion).
        if (settling_time < 0.0 && relative_error <= kSettlingThreshold) {
            bool stays_settled = true;
            for (size_t j = i; j < results.size(); ++j) {
                double later_error = std::abs(results[j].Measured - setpoint);
                double later_rel = (std::abs(setpoint) > 1e-6) ? later_error / std::abs(setpoint) : later_error;
                if (later_rel > kSettlingThreshold) { stays_settled = false; break; }
            }
            if (stays_settled) settling_time = results[i].time;
        }

        max_error = std::max(max_error, error);
        sum_squared_error += error * error;
    }

    double rms_error = std::sqrt(sum_squared_error / results.size());

    // Overshoot: how far peak output exceeds the setpoint, as a percentage
    double max_value = initial_value;
    for (const auto& pt : results) max_value = std::max(max_value, pt.Measured);
    double overshoot = (max_value > setpoint && std::abs(setpoint) > 1e-6)
                       ? ((max_value - setpoint) / std::abs(setpoint) * 100.0)
                       : 0.0;

    // Rise time: time from 10% to 90% of the full step change
    double target_change = setpoint - initial_value;
    double rise_time_10 = -1.0, rise_time_90 = -1.0;
    if (std::abs(target_change) > 1e-6) {
        for (size_t i = 0; i < results.size(); ++i) {
            double pct = (results[i].Measured - initial_value) / target_change;
            if (rise_time_10 < 0.0 && pct >= 0.1) rise_time_10 = results[i].time;
            if (rise_time_90 < 0.0 && pct >= 0.9) { rise_time_90 = results[i].time; break; }
        }
    }

    std::cout << "\n=== PID Performance Analysis ===\n"
              << "Initial value      : " << initial_value << " V\n"
              << "Final value        : " << final_value << " V\n"
              << "Steady-state error : " << steady_state_error << " V"
              << "  (how far from target after settling)\n";

    if (rise_time_10 >= 0.0 && rise_time_90 >= 0.0)
        std::cout << "Rise time (10-90%) : " << (rise_time_90 - rise_time_10) << " s"
                  << "  (speed of initial response)\n";
    else
        std::cout << "Rise time          : N/A\n";

    if (settling_time >= 0.0)
        std::cout << "Settling time (5%) : " << settling_time << " s"
                  << "  (time to stay within 5% of target)\n";
    else
        std::cout << "Settling time      : N/A\n";

    std::cout << "Maximum error      : " << max_error << " V"
              << "  (worst deviation from target)\n"
              << "RMS error          : " << rms_error << " V"
              << "  (overall tracking quality)\n"
              << "Overshoot          : " << overshoot << "%"
              << "  (peak excursion beyond target)\n";
}
