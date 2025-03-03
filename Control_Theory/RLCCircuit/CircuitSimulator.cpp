/**
 * @file CircuitSimulator.cpp
 * @brief Implementation of CircuitSimulator
 */

#include "CircuitSimulator.hpp"

CircuitSimulator::CircuitSimulator(std::shared_ptr<PIDController> pid,
                                   std::shared_ptr<CircuitModel> circuit)
    : pid_controller_(pid), circuit_(circuit),
      is_running_(false), current_setpoint_(0.0) {
    if (!pid) throw std::invalid_argument("PID controller pointer cannot be null");
    if (!circuit) throw std::invalid_argument("Circuit pointer cannot be null");
}

CircuitSimulator::~CircuitSimulator() {
    StopContinuousSimulation();
}

CircuitSimulator::SimulationResult CircuitSimulator::Run(double setpoint, double duration, double dt) {
    if (dt <= 0.0) throw std::invalid_argument("Time step must be positive");
    if (duration <= 0.0) throw std::invalid_argument("Duration must be positive");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_running_) throw std::runtime_error("Simulation is already running");
    }

    pid_controller_->Reset();
    circuit_->Reset();

    SimulationResult result;
    size_t num_steps = static_cast<size_t>(duration / dt) + 1;
    result.time.reserve(num_steps);
    result.setpoint.reserve(num_steps);
    result.output.reserve(num_steps);
    result.control_signal.reserve(num_steps);

    double current_time = 0.0;
    while (current_time <= duration) {
        double current_voltage = circuit_->GetVoltage();
        if (std::isnan(current_voltage) || std::isinf(current_voltage)) {
            current_voltage = 0.0;
        }

        double control_output = pid_controller_->Compute(setpoint, current_voltage, dt);
        circuit_->SimulateStep(control_output, dt);

        double updated_voltage = circuit_->GetVoltage();

        result.time.push_back(current_time);
        result.setpoint.push_back(setpoint);
        result.output.push_back(updated_voltage);
        result.control_signal.push_back(control_output);

        current_time += dt;
    }
    return result;
}

std::vector<CircuitSimulator::DataPoint> CircuitSimulator::RunSimulation(
    double setpoint, double duration, double time_step, DataCallback callback) {
    if (time_step <= 0.0) throw std::invalid_argument("Time step must be positive");
    if (duration <= 0.0) throw std::invalid_argument("Duration must be positive");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_running_) throw std::runtime_error("Simulation is already running");
    }

    pid_controller_->Reset();

    std::vector<DataPoint> results;
    results.reserve(static_cast<size_t>(duration / time_step) + 1);

    double current_time = 0.0;
    while (current_time <= duration) {
        DataPoint dp = RunSimulationStep(setpoint, current_time, time_step);
        results.push_back(dp);
        if (callback) callback(dp);
        current_time += time_step;
    }
    return results;
}

void CircuitSimulator::StartContinuousSimulation(double setpoint, double time_step,
                                                  DataCallback callback) {
    if (time_step <= 0.0) throw std::invalid_argument("Time step must be positive");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_running_) throw std::runtime_error("Simulation is already running");
        is_running_ = true;
        current_setpoint_ = setpoint;
    }
    pid_controller_->Reset();
    simulation_thread_ = std::thread(&CircuitSimulator::ContinuousSimulationThread,
                                     this, setpoint, time_step, callback);
}

void CircuitSimulator::StopContinuousSimulation() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!is_running_) return;
        is_running_ = false;
    }
    if (simulation_thread_.joinable()) simulation_thread_.join();
}

bool CircuitSimulator::IsSimulationRunning() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_running_;
}

void CircuitSimulator::ChangeSetpoint(double new_setpoint) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_setpoint_ = new_setpoint;
}

void CircuitSimulator::AddDisturbance(double magnitude) {
    circuit_->AddDisturbance(magnitude);
}

CircuitSimulator::DataPoint CircuitSimulator::RunSimulationStep(
    double setpoint, double current_time, double time_step) {
    double current_voltage = circuit_->GetVoltage();
    if (std::isnan(current_voltage) || std::isinf(current_voltage)) {
        current_voltage = 0.0;
    }

    double control_output = pid_controller_->Compute(setpoint, current_voltage, time_step);
    circuit_->SimulateStep(control_output, time_step);

    double updated_voltage = circuit_->GetVoltage();

    DataPoint dp;
    dp.time = current_time;
    dp.Setpoint = setpoint;
    dp.Measured = updated_voltage;
    dp.Control = control_output;
    dp.PTerm = pid_controller_->GetProportionalTerm();
    dp.ITerm = pid_controller_->GetIntegralTerm();
    dp.DTerm = pid_controller_->GetDerivativeTerm();
    dp.Saturated = pid_controller_->IsSaturated();
    return dp;
}

void CircuitSimulator::ContinuousSimulationThread(double setpoint, double time_step,
                                                   DataCallback callback) {
    double current_time = 0.0;
    double local_setpoint = setpoint;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!is_running_) break;
            local_setpoint = current_setpoint_;
        }
        DataPoint dp = RunSimulationStep(local_setpoint, current_time, time_step);
        if (callback) callback(dp);
        current_time += time_step;
        if (time_step > 0.01) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(time_step * 1000)));
        }
    }
}
