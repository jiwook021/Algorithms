/**
 * @file RCCircuit.hpp
 * @brief PID-controlled RC circuit simulation
 * @details Implements a PID controller (PIDController), a first-order RC circuit
 *          model (ElectricCircuit), and a closed-loop simulator (CircuitSimulator).
 *          Supports step response, disturbance rejection, and setpoint tracking.
 *
 * Time Complexity: O(N) where N is the number of simulation steps
 * Space Complexity: O(N) for result storage
 */

#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <mutex>
#include <memory>
#include <functional>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

/**
 * @class PIDController
 * @brief A thread-safe PID controller implementation with anti-windup protection
 * 
 * This class implements a Proportional-Integral-Derivative controller
 * that can be used to control various parameters in electric circuits
 * such as voltage, current, temperature, etc.
 */

class PIDController {
public:
    /**
     * @brief Construct a new PIDController object with specified gains
     * 
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param kd Derivative gain
     * @param outputMin Minimum output value
     * @param outputMax Maximum output value
     * @throws std::invalid_argument if gains are negative or limits are invalid
     */
    PIDController(double Kp, double Ki, double Kd, double OutputMin, double OutputMax)
        : Kp_(Kp), Ki_(Ki), Kd_(Kd), OutputMin_(OutputMin), OutputMax_(OutputMax),
          LastError_(0.0), IntegralSum_(0.0),
          ProportionalTerm_(0.0), IntegralTerm_(0.0), DerivativeTerm_(0.0),
          FirstCompute_(true), is_saturated_(false) {
        
        // Validate parameters
        if (Kp < 0.0 || Ki < 0.0 || Kd < 0.0) {
            throw std::invalid_argument("PID gains must be non-negative");
        }
        
        if (OutputMin >= OutputMax) {
            throw std::invalid_argument("Output minimum must be less than maximum");
        }
    }
    
    /**
     * @brief Compute the control output based on the current error
     * 
     * @param setpoint The desired target value
     * @param processVariable The current measured value
     * @param dt Time delta in seconds since last computation (defaults to auto calculation)
     * @return double The computed output value
     * @throws std::invalid_argument if dt is negative or zero when explicitly provided
     */
    double Compute(double Setpoint, double ProcessVariable, std::optional<double> Dt = std::nullopt) {
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock mutex to ensure thread safety
        
        // Calculate error
        double Error = Setpoint - ProcessVariable;
        
        // Calculate time delta
        double TimeDelta;
        if (Dt.has_value()) {
            TimeDelta = Dt.value();
            if (TimeDelta <= 0.0) {
                throw std::invalid_argument("Time delta must be positive");
            }
        } else {
            auto Now = std::chrono::high_resolution_clock::now();
            if (FirstCompute_) {
                TimeDelta = 0.0;
                FirstCompute_ = false;
            } else {
                TimeDelta = std::chrono::duration<double>(Now - LastTime_).count();
            }
            LastTime_ = Now;
        }
        
        // Calculate proportional term - proportional to current error
        ProportionalTerm_ = Kp_ * Error;
        
        // --- I term: accumulated error over time ---
        // Only accumulate when not saturated; otherwise the integral
        // grows unbounded while the actuator can't respond ("windup").
        if (TimeDelta > 0.0) {
            if (!is_saturated_) {
                // Accumulate integral sum (error × time)
                IntegralSum_ += Error * TimeDelta;
            }
            IntegralTerm_ = Ki_ * IntegralSum_;
        }
        
        // Calculate derivative term only if time has passed
        if (TimeDelta > 0.0) {
            // Use filtered derivative to reduce noise sensitivity
            // Rate of change of error over time
            DerivativeTerm_ = Kd_ * (Error - LastError_) / TimeDelta;
        } else {
            DerivativeTerm_ = 0.0;
        }
        
        // Save current error for next iteration
        LastError_ = Error;
        
        // Calculate total output
        double Output = ProportionalTerm_ + IntegralTerm_ + DerivativeTerm_;
        
        // Clamp to actuator limits
        double LimitedOutput = ApplyLimits(Output);

        // Saturation detection: output is clamped AND the error is still
        // pushing in the same direction — the actuator cannot follow.
        if ((LimitedOutput >= OutputMax_ && Error > 0) || 
            (LimitedOutput <= OutputMin_ && Error < 0)) {
            is_saturated_ = true;
        } else {
            is_saturated_ = false;
        }
        
        return LimitedOutput;
    }
    
    /**
     * @brief Reset the controller's integral and derivative terms
     */
    void Reset() {
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety
        
        LastError_ = 0.0;
        IntegralSum_ = 0.0;
        ProportionalTerm_ = 0.0;
        IntegralTerm_ = 0.0;
        DerivativeTerm_ = 0.0;
        FirstCompute_ = true;
        is_saturated_ = false;
    }
    
    /**
     * @brief Set new PID gains
     * 
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param kd Derivative gain
     * @throws std::invalid_argument if any gain is negative
     */
    void SetGains(double Kp, double Ki, double Kd) {
        // Validate parameters
        if (Kp < 0.0 || Ki < 0.0 || Kd < 0.0) {
            throw std::invalid_argument("PID gains must be non-negative");
        }
        
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety
        
        Kp_ = Kp;
        Ki_ = Ki;
        Kd_ = Kd;
    }
    
    /**
     * @brief Set output limits
     * 
     * @param min Minimum output value
     * @param max Maximum output value
     * @throws std::invalid_argument if min >= max
     */
    void SetOutputLimits(double min, double max) {
        // Validate parameters
        if (min >= max) {
            throw std::invalid_argument("Output minimum must be less than maximum");
        }
        
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety
        
        OutputMin_ = min;
        OutputMax_ = max;
    }
    
    /**
     * @brief Get the last computed error
     * 
     * @return double The last error value
     */
    double GetLastError() const {
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety for reading
        return LastError_;
    }
    
    /**
     * @brief Get the current proportional term contribution
     * 
     * @return double The proportional term
     */
    double GetProportionalTerm() const {
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety for reading
        return ProportionalTerm_;
    }
    
    /**
     * @brief Get the current integral term contribution
     * 
     * @return double The integral term
     */
    double GetIntegralTerm() const {
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety for reading
        return IntegralTerm_;
    }
    
    /**
     * @brief Get the current derivative term contribution
     * 
     * @return double The derivative term
     */
    double GetDerivativeTerm() const {
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety for reading
        return DerivativeTerm_;
    }
    
    /**
     * @brief Check if anti-windup protection is currently active
     * 
     * @return bool True if output is saturated (clamped at limits while error persists)
     */
    bool IsSaturated() const {
        std::lock_guard<std::mutex> Lock(Mutex_);
        return is_saturated_;
    }

private:
    // PID gains
    double Kp_; // Proportional gain
    double Ki_; // Integral gain
    double Kd_; // Derivative gain
    
    // Output limits
    double OutputMin_; // Minimum output value
    double OutputMax_; // Maximum output value
    
    // State variables
    double LastError_;      // Last error value for derivative calculation
    double IntegralSum_;    // Sum of errors for integral calculation
    double ProportionalTerm_; // Last proportional term
    double IntegralTerm_;     // Last integral term
    double DerivativeTerm_;   // Last derivative term
    
    // Timing
    std::chrono::time_point<std::chrono::high_resolution_clock> LastTime_; // Last computation time
    bool FirstCompute_; // Flag to indicate first computation
    
    // Saturation state: true when output is clamped and error pushes further
    bool is_saturated_;
    
    // Thread safety
    mutable std::mutex Mutex_; // Mutex for thread safety on all operations
    
    /**
     * @brief Apply output limits to a value
     * 
     * @param value The value to constrain
     * @return double The constrained value
     */
    double ApplyLimits(double value) const {
        // Constrain the output value between min and max limits
        return std::clamp(value, OutputMin_, OutputMax_);
    }
};

/**
 * @class ElectricCircuit
 * @brief A simple electric circuit simulator for demonstrating PID control
 * 
 * This class simulates a basic electric circuit with voltage, current,
 * and resistance. It includes noise and disturbances to demonstrate
 * the effectiveness of PID control.
 */
class ElectricCircuit {
public:
    /**
     * @brief Construct a new ElectricCircuit object
     * 
     * @param initialVoltage Initial voltage value in volts
     * @param resistance Circuit resistance in ohms
     * @param capacitance Circuit capacitance in farads
     * @param noiseLevel Level of noise in the circuit (0.0 to 1.0)
     * @throws std::invalid_argument if parameters are invalid
     */
    ElectricCircuit(double InitialVoltage, double Resistance, double Capacitance, double NoiseLevel)
        : Voltage_(InitialVoltage), Current_(InitialVoltage / Resistance), 
          Resistance_(Resistance), Capacitance_(Capacitance), NoiseLevel_(NoiseLevel) {
        
        // Validate parameters
        if (Resistance <= 0.0) {
            throw std::invalid_argument("Resistance must be positive");
        }
        
        if (Capacitance <= 0.0) {
            throw std::invalid_argument("Capacitance must be positive");
        }
        
        if (NoiseLevel < 0.0 || NoiseLevel > 1.0) {
            throw std::invalid_argument("Noise level must be between 0.0 and 1.0");
        }
        
        // Initialize random number generator with a random seed
        unsigned Seed = std::chrono::system_clock::now().time_since_epoch().count();
        Rng_ = std::mt19937(Seed);
        
        // Initialize noise distribution - zero mean, unit variance
        NoiseDist_ = std::normal_distribution<double>(0.0, 1.0);
    }
    
    /**
     * @brief Apply an input voltage to the circuit
     * 
     * @param voltage Input voltage in volts
     * @param dt Time step in seconds
     * @throws std::invalid_argument if dt is negative or zero
     */
    void ApplyVoltage(double Voltage, double Dt) {
        if (Dt <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }
        
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety
        
        // Update circuit state with the new voltage
        UpdateCircuitState(Voltage, Dt);
    }
    
    /**
     * @brief Get the current voltage across the circuit
     * 
     * @return double Current voltage in volts
     */
    double GetVoltage() const {
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety for reading
        return Voltage_;
    }
    
    /**
     * @brief Get the current flowing through the circuit
     * 
     * @return double Current in amperes
     */
    double GetCurrent() const {
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety for reading
        return Current_;
    }
    
    /**
     * @brief Add a disturbance to the circuit (e.g. load change)
     * 
     * @param magnitude Magnitude of the disturbance (positive or negative)
     */
    void AddDisturbance(double Magnitude) {
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety
        
        // Add a sudden change to the voltage to simulate a disturbance
        Voltage_ += Magnitude;
    }
    
    /**
     * @brief Set the noise level in the circuit
     * 
     * @param level Noise level from 0.0 to 1.0
     * @throws std::invalid_argument if level is outside valid range
     */
    void SetNoiseLevel(double Level) {
        if (Level < 0.0 || Level > 1.0) {
            throw std::invalid_argument("Noise level must be between 0.0 and 1.0");
        }
        
        std::lock_guard<std::mutex> Lock(Mutex_); // Lock for thread safety
        NoiseLevel_ = Level;
    }

private:
    double Voltage_;      // Circuit voltage (V)
    double Current_;      // Circuit current (A)
    double Resistance_;   // Circuit resistance (Ω)
    double Capacitance_;  // Circuit capacitance (F)
    double NoiseLevel_;   // Noise level (0.0 to 1.0)
    
    // Random number generation for noise
    std::mt19937 Rng_; // Random number generator
    std::normal_distribution<double> NoiseDist_; // Normal distribution for noise
    
    // Thread safety
    mutable std::mutex Mutex_; // Mutex for thread safety
    
    /**
     * @brief Generate noise for the circuit
     * 
     * @return double Noise value based on current noise level
     */
    double GenerateNoise() {
        // Generate random noise based on the current noise level
        return NoiseLevel_ * NoiseDist_(Rng_);
    }
    
    /**
     * @brief Update circuit state based on circuit equations
     * 
     * @param inputVoltage Input voltage
     * @param dt Time step
     */
    void UpdateCircuitState(double InputVoltage, double Dt) {
        // Simple RC circuit differential equation: dV/dt = (V_in - V_out)/(R*C)
        // Using Euler integration method for simplicity
        double VoltageChange = (InputVoltage - Voltage_) / (Resistance_ * Capacitance_) * Dt;
        
        // Add noise to simulate real-world conditions
        VoltageChange += GenerateNoise();
        
        // Update voltage
        Voltage_ += VoltageChange;
        
        // Update current based on Ohm's law: I = V/R
        Current_ = Voltage_ / Resistance_;
    }
};

/**
 * @class CircuitSimulator
 * @brief A simulator for PID control of an electric circuit
 * 
 * This class combines the PID controller and electric circuit models
 * to demonstrate closed-loop control of circuit parameters like voltage
 * or current.
 */
class CircuitSimulator {
public:
    /**
     * @brief Type definition for simulation data point
     * Contains time, setpoint, measured value, and control output
     */
    struct DataPoint {
        double time;      // Simulation time in seconds
        double Setpoint;  // Target value
        double Measured;  // Measured value
        double Control;   // Control signal
        double PTerm;    // Proportional term
        double ITerm;    // Integral term
        double DTerm;    // Derivative term
        bool Saturated; // True if controller output is at its limits
    };
    
    /**
     * @brief Type definition for a data callback function
     * Called with each new data point during simulation
     */
    using DataCallback = std::function<void(const DataPoint&)>;
    
    /**
     * @brief Construct a new CircuitSimulator for voltage control
     * 
     * @param pidController Shared pointer to a PID controller
     * @param circuit Shared pointer to an electric circuit
     * @throws std::invalid_argument if pointers are null
     */
    CircuitSimulator(std::shared_ptr<PIDController> PidController, 
                    std::shared_ptr<ElectricCircuit> Circuit)
        : PidController_(PidController), Circuit_(Circuit), IsRunning_(false), CurrentSetpoint_(0.0) {
        
        // Validate parameters
        if (!PidController) {
            throw std::invalid_argument("PID controller pointer cannot be null");
        }
        
        if (!Circuit) {
            throw std::invalid_argument("Circuit pointer cannot be null");
        }
    }
    
    /**
     * @brief Destroy the CircuitSimulator object
     * Ensures simulation is stopped
     */
    ~CircuitSimulator() {
        // Ensure simulation is stopped before destruction to prevent dangling threads
        StopContinuousSimulation();
    }
    
    /**
     * @brief Run the simulation for a specified duration
     * 
     * @param setpoint The target value to maintain
     * @param duration Simulation duration in seconds
     * @param timeStep Time step in seconds
     * @param callback Optional callback for real-time data
     * @return std::vector<DataPoint> Simulation results
     * @throws std::invalid_argument if parameters are invalid
     * @throws std::runtime_error if simulation is already running
     */
    std::vector<DataPoint> RunSimulation(double Setpoint, double Duration, 
                                        double TimeStep, DataCallback Callback = nullptr) {
        
        // Validate parameters
        if (TimeStep <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }
        
        if (Duration <= 0.0) {
            throw std::invalid_argument("Duration must be positive");
        }
        
        // Check if a simulation is already running
        {
            std::lock_guard<std::mutex> Lock(Mutex_);
            if (IsRunning_) {
                throw std::runtime_error("Simulation is already running");
            }
        }
        
        // Reset PID controller to clear previous state
        PidController_->Reset();
        
        // Initialize results vector with capacity to avoid reallocations
        std::vector<DataPoint> Results;
        Results.reserve(static_cast<size_t>(Duration / TimeStep) + 1);
        
        // Run simulation for specified duration
        double CurrentTime = 0.0;
        while (CurrentTime <= Duration) {
            DataPoint DataPoint = RunSimulationStep(Setpoint, CurrentTime, TimeStep);
            
            // Store data point
            Results.push_back(DataPoint);
            
            // Call callback if provided
            if (Callback) {
                Callback(DataPoint);
            }
            
            // Increment time
            CurrentTime += TimeStep;
        }
        
        return Results;
    }
    
    /**
     * @brief Start a continuous simulation
     * 
     * @param setpoint The target value to maintain
     * @param timeStep Time step in seconds
     * @param callback Callback for real-time data
     * @throws std::invalid_argument if parameters are invalid
     * @throws std::runtime_error if simulation is already running
     */
    void StartContinuousSimulation(double Setpoint, double TimeStep, DataCallback Callback) {
        // Validate parameters
        if (TimeStep <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }
        
        // Acquire lock to check and update simulation state
        {
            std::lock_guard<std::mutex> Lock(Mutex_);
            
            if (IsRunning_) {
                throw std::runtime_error("Simulation is already running");
            }
            
            IsRunning_ = true;
            CurrentSetpoint_ = Setpoint;
        }
        
        // Reset PID controller
        PidController_->Reset();
        
        // Start simulation in a separate thread
        SimulationThread_ = std::thread(&CircuitSimulator::ContinuousSimulationThread, 
                                      this, Setpoint, TimeStep, Callback);
    }
    
    /**
     * @brief Stop the continuous simulation
     */
    void StopContinuousSimulation() {
        // Set running flag to false - signals the thread to stop
        {
            std::lock_guard<std::mutex> Lock(Mutex_);
            if (!IsRunning_) {
                return; // Already stopped
            }
            IsRunning_ = false;
        }
        
        // Wait for simulation thread to finish if it's running
        if (SimulationThread_.joinable()) {
            SimulationThread_.join();
        }
    }
    
    /**
     * @brief Check if a simulation is currently running
     * 
     * @return true if simulation is running
     * @return false if simulation is not running
     */
    bool IsSimulationRunning() const {
        std::lock_guard<std::mutex> Lock(Mutex_);
        return IsRunning_;
    }
    
    /**
     * @brief Change the setpoint during a continuous simulation
     * 
     * @param newSetpoint The new target value
     */
    void ChangeSetpoint(double NewSetpoint) {
        std::lock_guard<std::mutex> Lock(Mutex_);
        CurrentSetpoint_ = NewSetpoint;
    }
    
    /**
     * @brief Add a disturbance to the circuit during simulation
     * 
     * @param magnitude Magnitude of the disturbance
     */
    void AddDisturbance(double Magnitude) {
        Circuit_->AddDisturbance(Magnitude);
    }

private:
    std::shared_ptr<PIDController> PidController_; // PID controller
    std::shared_ptr<ElectricCircuit> Circuit_;     // Electric circuit
    
    // Continuous simulation state
    bool IsRunning_;            // Flag indicating if simulation is running
    double CurrentSetpoint_;    // Current setpoint value
    std::thread SimulationThread_; // Thread for continuous simulation
    
    // Thread safety
    mutable std::mutex Mutex_; // Mutex for thread safety
    
    /**
     * @brief Run a single simulation step
     * 
     * @param setpoint Current setpoint
     * @param currentTime Current simulation time
     * @param timeStep Time step
     * @return DataPoint Data from this simulation step
     */
    DataPoint RunSimulationStep(double Setpoint, double CurrentTime, double TimeStep) {
        // Get current circuit voltage
        double CurrentVoltage = Circuit_->GetVoltage();
        
        // Compute control output using PID controller
        double ControlOutput = PidController_->Compute(Setpoint, CurrentVoltage, TimeStep);
        
        // Apply control output to circuit
        Circuit_->ApplyVoltage(ControlOutput, TimeStep);
        
        // Create data point with current values
        DataPoint DataPoint;
        DataPoint.time = CurrentTime;
        DataPoint.Setpoint = Setpoint;
        DataPoint.Measured = CurrentVoltage;
        DataPoint.Control = ControlOutput;
        DataPoint.PTerm = PidController_->GetProportionalTerm();
        DataPoint.ITerm = PidController_->GetIntegralTerm();
        DataPoint.DTerm = PidController_->GetDerivativeTerm();
        DataPoint.Saturated = PidController_->IsSaturated();
        
        return DataPoint;
    }
    
    /**
     * @brief Thread function for continuous simulation
     * 
     * @param setpoint Initial setpoint
     * @param timeStep Time step
     * @param callback Data callback function
     */
    void ContinuousSimulationThread(double Setpoint, double TimeStep, DataCallback Callback) {
        double CurrentTime = 0.0;
        double LocalSetpoint = Setpoint;
        
        // Run simulation until stopped
        while (true) {
            // Check if simulation should continue
            {
                std::lock_guard<std::mutex> Lock(Mutex_);
                if (!IsRunning_) {
                    break;  // Exit the loop if simulation is stopped
                }
                LocalSetpoint = CurrentSetpoint_;  // Get the latest setpoint
            }
            
            // Run single simulation step
            DataPoint DataPoint = RunSimulationStep(LocalSetpoint, CurrentTime, TimeStep);
            
            // Call callback with data
            if (Callback) {
                Callback(DataPoint);
            }
            
            // Increment time
            CurrentTime += TimeStep;
            
            // sleep to maintain real-time simulation if timeStep is large enough
            // This prevents the simulation from running too fast
            if (TimeStep > 0.01) {
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<int>(TimeStep * 1000)));
            }
        }
    }
};

// Function to print simulation results to console
