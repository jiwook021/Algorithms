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
    PIDController(double kp, double ki, double kd, double outputMin, double outputMax)
        : kp_(kp), ki_(ki), kd_(kd), outputMin_(outputMin), outputMax_(outputMax),
          lastError_(0.0), integralSum_(0.0),
          proportionalTerm_(0.0), integralTerm_(0.0), derivativeTerm_(0.0),
          firstCompute_(true), antiWindupActive_(false) {
        
        // Validate parameters
        if (kp < 0.0 || ki < 0.0 || kd < 0.0) {
            throw std::invalid_argument("PID gains must be non-negative");
        }
        
        if (outputMin >= outputMax) {
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
    double compute(double setpoint, double processVariable, std::optional<double> dt = std::nullopt) {
        std::lock_guard<std::mutex> lock(mutex_); // Lock mutex to ensure thread safety
        
        // Calculate error
        double error = setpoint - processVariable;
        
        // Calculate time delta
        double timeDelta;
        if (dt.has_value()) {
            timeDelta = dt.value();
            if (timeDelta <= 0.0) {
                throw std::invalid_argument("Time delta must be positive");
            }
        } else {
            auto now = std::chrono::high_resolution_clock::now();
            if (firstCompute_) {
                timeDelta = 0.0;
                firstCompute_ = false;
            } else {
                timeDelta = std::chrono::duration<double>(now - lastTime_).count();
            }
            lastTime_ = now;
        }
        
        // Calculate proportional term - proportional to current error
        proportionalTerm_ = kp_ * error;
        
        // Calculate integral term only if time has passed
        if (timeDelta > 0.0) {
            // Only accumulate integral if not in anti-windup state
            if (!antiWindupActive_) {
                // Accumulate integral sum (error × time)
                integralSum_ += error * timeDelta;
            }
            integralTerm_ = ki_ * integralSum_;
        }
        
        // Calculate derivative term only if time has passed
        if (timeDelta > 0.0) {
            // Use filtered derivative to reduce noise sensitivity
            // Rate of change of error over time
            derivativeTerm_ = kd_ * (error - lastError_) / timeDelta;
        } else {
            derivativeTerm_ = 0.0;
        }
        
        // Save current error for next iteration
        lastError_ = error;
        
        // Calculate total output
        double output = proportionalTerm_ + integralTerm_ + derivativeTerm_;
        
        // Apply output limits and check for anti-windup condition
        double limitedOutput = applyLimits(output);
        
        // Check for anti-windup: if output is saturated and error is in the same direction
        // This means that integral action would push the output further into saturation
        if ((limitedOutput >= outputMax_ && error > 0) || 
            (limitedOutput <= outputMin_ && error < 0)) {
            antiWindupActive_ = true;
        } else {
            antiWindupActive_ = false;
        }
        
        return limitedOutput;
    }
    
    /**
     * @brief Reset the controller's integral and derivative terms
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety
        
        lastError_ = 0.0;
        integralSum_ = 0.0;
        proportionalTerm_ = 0.0;
        integralTerm_ = 0.0;
        derivativeTerm_ = 0.0;
        firstCompute_ = true;
        antiWindupActive_ = false;
    }
    
    /**
     * @brief Set new PID gains
     * 
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param kd Derivative gain
     * @throws std::invalid_argument if any gain is negative
     */
    void setGains(double kp, double ki, double kd) {
        // Validate parameters
        if (kp < 0.0 || ki < 0.0 || kd < 0.0) {
            throw std::invalid_argument("PID gains must be non-negative");
        }
        
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety
        
        kp_ = kp;
        ki_ = ki;
        kd_ = kd;
    }
    
    /**
     * @brief Set output limits
     * 
     * @param min Minimum output value
     * @param max Maximum output value
     * @throws std::invalid_argument if min >= max
     */
    void setOutputLimits(double min, double max) {
        // Validate parameters
        if (min >= max) {
            throw std::invalid_argument("Output minimum must be less than maximum");
        }
        
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety
        
        outputMin_ = min;
        outputMax_ = max;
    }
    
    /**
     * @brief Get the last computed error
     * 
     * @return double The last error value
     */
    double getLastError() const {
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety for reading
        return lastError_;
    }
    
    /**
     * @brief Get the current proportional term contribution
     * 
     * @return double The proportional term
     */
    double getProportionalTerm() const {
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety for reading
        return proportionalTerm_;
    }
    
    /**
     * @brief Get the current integral term contribution
     * 
     * @return double The integral term
     */
    double getIntegralTerm() const {
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety for reading
        return integralTerm_;
    }
    
    /**
     * @brief Get the current derivative term contribution
     * 
     * @return double The derivative term
     */
    double getDerivativeTerm() const {
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety for reading
        return derivativeTerm_;
    }
    
    /**
     * @brief Check if anti-windup protection is currently active
     * 
     * @return bool True if anti-windup is active, false otherwise
     */
    bool isAntiWindupActive() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return antiWindupActive_;
    }

private:
    // PID gains
    double kp_; // Proportional gain
    double ki_; // Integral gain
    double kd_; // Derivative gain
    
    // Output limits
    double outputMin_; // Minimum output value
    double outputMax_; // Maximum output value
    
    // State variables
    double lastError_;      // Last error value for derivative calculation
    double integralSum_;    // Sum of errors for integral calculation
    double proportionalTerm_; // Last proportional term
    double integralTerm_;     // Last integral term
    double derivativeTerm_;   // Last derivative term
    
    // Timing
    std::chrono::time_point<std::chrono::high_resolution_clock> lastTime_; // Last computation time
    bool firstCompute_; // Flag to indicate first computation
    
    // Anti-windup state
    bool antiWindupActive_; // Flag to indicate if anti-windup protection is active
    
    // Thread safety
    mutable std::mutex mutex_; // Mutex for thread safety on all operations
    
    /**
     * @brief Apply output limits to a value
     * 
     * @param value The value to constrain
     * @return double The constrained value
     */
    double applyLimits(double value) const {
        // Constrain the output value between min and max limits
        return std::clamp(value, outputMin_, outputMax_);
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
    ElectricCircuit(double initialVoltage, double resistance, double capacitance, double noiseLevel)
        : voltage_(initialVoltage), current_(initialVoltage / resistance), 
          resistance_(resistance), capacitance_(capacitance), noiseLevel_(noiseLevel) {
        
        // Validate parameters
        if (resistance <= 0.0) {
            throw std::invalid_argument("Resistance must be positive");
        }
        
        if (capacitance <= 0.0) {
            throw std::invalid_argument("Capacitance must be positive");
        }
        
        if (noiseLevel < 0.0 || noiseLevel > 1.0) {
            throw std::invalid_argument("Noise level must be between 0.0 and 1.0");
        }
        
        // Initialize random number generator with a random seed
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        rng_ = std::mt19937(seed);
        
        // Initialize noise distribution - zero mean, unit variance
        noiseDist_ = std::normal_distribution<double>(0.0, 1.0);
    }
    
    /**
     * @brief Apply an input voltage to the circuit
     * 
     * @param voltage Input voltage in volts
     * @param dt Time step in seconds
     * @throws std::invalid_argument if dt is negative or zero
     */
    void applyVoltage(double voltage, double dt) {
        if (dt <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }
        
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety
        
        // Update circuit state with the new voltage
        updateCircuitState(voltage, dt);
    }
    
    /**
     * @brief Get the current voltage across the circuit
     * 
     * @return double Current voltage in volts
     */
    double getVoltage() const {
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety for reading
        return voltage_;
    }
    
    /**
     * @brief Get the current flowing through the circuit
     * 
     * @return double Current in amperes
     */
    double getCurrent() const {
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety for reading
        return current_;
    }
    
    /**
     * @brief Add a disturbance to the circuit (e.g. load change)
     * 
     * @param magnitude Magnitude of the disturbance (positive or negative)
     */
    void addDisturbance(double magnitude) {
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety
        
        // Add a sudden change to the voltage to simulate a disturbance
        voltage_ += magnitude;
    }
    
    /**
     * @brief Set the noise level in the circuit
     * 
     * @param level Noise level from 0.0 to 1.0
     * @throws std::invalid_argument if level is outside valid range
     */
    void setNoiseLevel(double level) {
        if (level < 0.0 || level > 1.0) {
            throw std::invalid_argument("Noise level must be between 0.0 and 1.0");
        }
        
        std::lock_guard<std::mutex> lock(mutex_); // Lock for thread safety
        noiseLevel_ = level;
    }

private:
    double voltage_;      // Circuit voltage (V)
    double current_;      // Circuit current (A)
    double resistance_;   // Circuit resistance (Ω)
    double capacitance_;  // Circuit capacitance (F)
    double noiseLevel_;   // Noise level (0.0 to 1.0)
    
    // Random number generation for noise
    std::mt19937 rng_; // Random number generator
    std::normal_distribution<double> noiseDist_; // Normal distribution for noise
    
    // Thread safety
    mutable std::mutex mutex_; // Mutex for thread safety
    
    /**
     * @brief Generate noise for the circuit
     * 
     * @return double Noise value based on current noise level
     */
    double generateNoise() {
        // Generate random noise based on the current noise level
        return noiseLevel_ * noiseDist_(rng_);
    }
    
    /**
     * @brief Update circuit state based on circuit equations
     * 
     * @param inputVoltage Input voltage
     * @param dt Time step
     */
    void updateCircuitState(double inputVoltage, double dt) {
        // Simple RC circuit differential equation: dV/dt = (V_in - V_out)/(R*C)
        // Using Euler integration method for simplicity
        double voltageChange = (inputVoltage - voltage_) / (resistance_ * capacitance_) * dt;
        
        // Add noise to simulate real-world conditions
        voltageChange += generateNoise();
        
        // Update voltage
        voltage_ += voltageChange;
        
        // Update current based on Ohm's law: I = V/R
        current_ = voltage_ / resistance_;
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
        double setpoint;  // Target value
        double measured;  // Measured value
        double control;   // Control signal
        double p_term;    // Proportional term
        double i_term;    // Integral term
        double d_term;    // Derivative term
        bool anti_windup; // Anti-windup state
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
    CircuitSimulator(std::shared_ptr<PIDController> pidController, 
                    std::shared_ptr<ElectricCircuit> circuit)
        : pidController_(pidController), circuit_(circuit), isRunning_(false), currentSetpoint_(0.0) {
        
        // Validate parameters
        if (!pidController) {
            throw std::invalid_argument("PID controller pointer cannot be null");
        }
        
        if (!circuit) {
            throw std::invalid_argument("Circuit pointer cannot be null");
        }
    }
    
    /**
     * @brief Destroy the CircuitSimulator object
     * Ensures simulation is stopped
     */
    ~CircuitSimulator() {
        // Ensure simulation is stopped before destruction to prevent dangling threads
        stopContinuousSimulation();
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
    std::vector<DataPoint> runSimulation(double setpoint, double duration, 
                                        double timeStep, DataCallback callback = nullptr) {
        
        // Validate parameters
        if (timeStep <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }
        
        if (duration <= 0.0) {
            throw std::invalid_argument("Duration must be positive");
        }
        
        // Check if a simulation is already running
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (isRunning_) {
                throw std::runtime_error("Simulation is already running");
            }
        }
        
        // Reset PID controller to clear previous state
        pidController_->reset();
        
        // Initialize results vector with capacity to avoid reallocations
        std::vector<DataPoint> results;
        results.reserve(static_cast<size_t>(duration / timeStep) + 1);
        
        // Run simulation for specified duration
        double currentTime = 0.0;
        while (currentTime <= duration) {
            DataPoint dataPoint = runSimulationStep(setpoint, currentTime, timeStep);
            
            // Store data point
            results.push_back(dataPoint);
            
            // Call callback if provided
            if (callback) {
                callback(dataPoint);
            }
            
            // Increment time
            currentTime += timeStep;
        }
        
        return results;
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
    void startContinuousSimulation(double setpoint, double timeStep, DataCallback callback) {
        // Validate parameters
        if (timeStep <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }
        
        // Acquire lock to check and update simulation state
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            if (isRunning_) {
                throw std::runtime_error("Simulation is already running");
            }
            
            isRunning_ = true;
            currentSetpoint_ = setpoint;
        }
        
        // Reset PID controller
        pidController_->reset();
        
        // Start simulation in a separate thread
        simulationThread_ = std::thread(&CircuitSimulator::continuousSimulationThread, 
                                      this, setpoint, timeStep, callback);
    }
    
    /**
     * @brief Stop the continuous simulation
     */
    void stopContinuousSimulation() {
        // Set running flag to false - signals the thread to stop
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!isRunning_) {
                return; // Already stopped
            }
            isRunning_ = false;
        }
        
        // Wait for simulation thread to finish if it's running
        if (simulationThread_.joinable()) {
            simulationThread_.join();
        }
    }
    
    /**
     * @brief Check if a simulation is currently running
     * 
     * @return true if simulation is running
     * @return false if simulation is not running
     */
    bool isSimulationRunning() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return isRunning_;
    }
    
    /**
     * @brief Change the setpoint during a continuous simulation
     * 
     * @param newSetpoint The new target value
     */
    void changeSetpoint(double newSetpoint) {
        std::lock_guard<std::mutex> lock(mutex_);
        currentSetpoint_ = newSetpoint;
    }
    
    /**
     * @brief Add a disturbance to the circuit during simulation
     * 
     * @param magnitude Magnitude of the disturbance
     */
    void addDisturbance(double magnitude) {
        circuit_->addDisturbance(magnitude);
    }

private:
    std::shared_ptr<PIDController> pidController_; // PID controller
    std::shared_ptr<ElectricCircuit> circuit_;     // Electric circuit
    
    // Continuous simulation state
    bool isRunning_;            // Flag indicating if simulation is running
    double currentSetpoint_;    // Current setpoint value
    std::thread simulationThread_; // Thread for continuous simulation
    
    // Thread safety
    mutable std::mutex mutex_; // Mutex for thread safety
    
    /**
     * @brief Run a single simulation step
     * 
     * @param setpoint Current setpoint
     * @param currentTime Current simulation time
     * @param timeStep Time step
     * @return DataPoint Data from this simulation step
     */
    DataPoint runSimulationStep(double setpoint, double currentTime, double timeStep) {
        // Get current circuit voltage
        double currentVoltage = circuit_->getVoltage();
        
        // Compute control output using PID controller
        double controlOutput = pidController_->compute(setpoint, currentVoltage, timeStep);
        
        // Apply control output to circuit
        circuit_->applyVoltage(controlOutput, timeStep);
        
        // Create data point with current values
        DataPoint dataPoint;
        dataPoint.time = currentTime;
        dataPoint.setpoint = setpoint;
        dataPoint.measured = currentVoltage;
        dataPoint.control = controlOutput;
        dataPoint.p_term = pidController_->getProportionalTerm();
        dataPoint.i_term = pidController_->getIntegralTerm();
        dataPoint.d_term = pidController_->getDerivativeTerm();
        dataPoint.anti_windup = pidController_->isAntiWindupActive();
        
        return dataPoint;
    }
    
    /**
     * @brief Thread function for continuous simulation
     * 
     * @param setpoint Initial setpoint
     * @param timeStep Time step
     * @param callback Data callback function
     */
    void continuousSimulationThread(double setpoint, double timeStep, DataCallback callback) {
        double currentTime = 0.0;
        double localSetpoint = setpoint;
        
        // Run simulation until stopped
        while (true) {
            // Check if simulation should continue
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!isRunning_) {
                    break;  // Exit the loop if simulation is stopped
                }
                localSetpoint = currentSetpoint_;  // Get the latest setpoint
            }
            
            // Run single simulation step
            DataPoint dataPoint = runSimulationStep(localSetpoint, currentTime, timeStep);
            
            // Call callback with data
            if (callback) {
                callback(dataPoint);
            }
            
            // Increment time
            currentTime += timeStep;
            
            // Sleep to maintain real-time simulation if timeStep is large enough
            // This prevents the simulation from running too fast
            if (timeStep > 0.01) {
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<int>(timeStep * 1000)));
            }
        }
    }
};

// Function to print simulation results to console
void printDataPoint(const CircuitSimulator::DataPoint& data) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time: " << std::setw(8) << data.time << "s | ";
    std::cout << "Setpoint: " << std::setw(8) << data.setpoint << "V | ";
    std::cout << "Measured: " << std::setw(8) << data.measured << "V | ";
    std::cout << "Control: " << std::setw(8) << data.control << "V | ";
    std::cout << "P: " << std::setw(8) << data.p_term << " | ";
    std::cout << "I: " << std::setw(8) << data.i_term << " | ";
    std::cout << "D: " << std::setw(8) << data.d_term << " | ";
    std::cout << "Anti-windup: " << (data.anti_windup ? "On" : "Off") << std::endl;
}

// Function to save simulation results to a CSV file
void saveResultsToCSV(const std::vector<CircuitSimulator::DataPoint>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write header
    file << "Time,Setpoint,Measured,Control,P_Term,I_Term,D_Term,Anti_Windup" << std::endl;
    
    // Write data
    for (const auto& data : results) {
        file << data.time << ","
             << data.setpoint << ","
             << data.measured << ","
             << data.control << ","
             << data.p_term << ","
             << data.i_term << ","
             << data.d_term << ","
             << (data.anti_windup ? 1 : 0) << std::endl;
    }
    
    std::cout << "Results saved to " << filename << std::endl;
}

// Analyze PID performance metrics from simulation results
void analyzePIDPerformance(const std::vector<CircuitSimulator::DataPoint>& results, double setpoint) {
    if (results.empty()) {
        std::cerr << "Error: No simulation results to analyze." << std::endl;
        return;
    }
    
    // Extract key metrics
    double finalValue = results.back().measured;
    double initialValue = results.front().measured;
    double settlingThreshold = 0.05; // 5% settling threshold
    double riseThreshold10 = 0.1;    // 10% rise threshold
    double riseThreshold90 = 0.9;    // 90% rise threshold
    
    // Calculate error metrics
    double maxError = 0.0;
    double sumSquaredError = 0.0;
    double steadyStateError = std::abs(finalValue - setpoint);
    
    // Find settling time (time to stay within 5% of final value)
    double settlingTime = results.back().time;
    bool settled = false;
    size_t settlingIndex = results.size() - 1;
    
    // Find when the system first reaches within 5% of setpoint (not final value)
    for (size_t i = 0; i < results.size(); ++i) {
        double error = std::abs(results[i].measured - setpoint);
        double relativeError = error / setpoint;
        
        if (relativeError <= settlingThreshold) {
            // Check if it stays within this band for the rest of the simulation
            bool staysSettled = true;
            for (size_t j = i; j < results.size(); ++j) {
                double laterError = std::abs(results[j].measured - setpoint) / setpoint;
                if (laterError > settlingThreshold) {
                    staysSettled = false;
                    break;
                }
            }
            
            if (staysSettled) {
                settled = true;
                settlingIndex = i;
                settlingTime = results[i].time;
                break;
            }
        }
    }
    
    if (!settled) {
        settlingTime = -1.0; // Indicates system never settled
    }
    
    // Find rise time (10% to 90% of step)
    double targetChange = setpoint - initialValue;
    double riseTime10 = -1.0;
    double riseTime90 = -1.0;
    bool found10 = false;
    bool found90 = false;
    
    for (size_t i = 0; i < results.size(); ++i) {
        double percentComplete = (results[i].measured - initialValue) / targetChange;
        
        // Find 10% rise point
        if (!found10 && percentComplete >= riseThreshold10) {
            riseTime10 = results[i].time;
            found10 = true;
        }
        
        // Find 90% rise point
        if (!found90 && percentComplete >= riseThreshold90) {
            riseTime90 = results[i].time;
            found90 = true;
            break;
        }
        
        // Track maximum error and sum squared error
        double error = std::abs(results[i].measured - setpoint);
        maxError = std::max(maxError, error);
        sumSquaredError += error * error;
    }
    
    // Calculate RMS error
    double rmsError = std::sqrt(sumSquaredError / results.size());
    
    // Check for overshoot
    double maxValue = initialValue;
    for (const auto& point : results) {
        maxValue = std::max(maxValue, point.measured);
    }
    double overshoot = (maxValue > setpoint) ? ((maxValue - setpoint) / setpoint * 100.0) : 0.0;
    
    // Print analysis
    std::cout << "\nPID Performance Analysis:" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "Initial value: " << initialValue << " V" << std::endl;
    std::cout << "Final value: " << finalValue << " V" << std::endl;
    std::cout << "Steady-state error: " << steadyStateError << " V (" 
              << (steadyStateError / setpoint * 100.0) << "%)" << std::endl;
    
    if (found10 && found90) {
        std::cout << "Rise time (10% to 90%): " << (riseTime90 - riseTime10) << " s" << std::endl;
    } else {
        std::cout << "Rise time: Not available - system did not cross both thresholds" << std::endl;
    }
    
    if (settled) {
        std::cout << "Settling time (±5%): " << settlingTime << " s" << std::endl;
    } else {
        std::cout << "Settling time: Not available - system did not settle within ±5%" << std::endl;
    }
    
    std::cout << "Maximum error: " << maxError << " V" << std::endl;
    std::cout << "RMS error: " << rmsError << " V" << std::endl;
    std::cout << "Overshoot: " << overshoot << "%" << std::endl;
}

int main() {
    try {
        std::cout << "PID Control Simulation for Electric Circuits" << std::endl;
        std::cout << "============================================" << std::endl;
        
        // Create PID controller with properly tuned parameters for the circuit
        // These values are adjusted to prevent oscillation
        auto pid = std::make_shared<PIDController>(2.4, 0.48, 0.16, 0.0, 12.0);
        
        // Create electric circuit with a slightly larger capacitance to slow down the dynamics
        // Time constant is now 0.5 seconds (RC = 10 * 0.05 = 0.5)
        auto circuit = std::make_shared<ElectricCircuit>(0.0, 10.0, 0.05, 0.005);
        
        // Create circuit simulator
        CircuitSimulator simulator(pid, circuit);
        
        // ===== Step Response Test =====
        // Run a step response simulation with a smaller time step (0.01s)
        double setpoint = 5.0;        // Target voltage: 5V
        double duration = 50.0;       // Simulation duration: 15 seconds
        double timeStep = 0.05;       // Time step: 0.01 seconds (10x smaller than before)
        
        std::cout << "\nRunning step response simulation..." << std::endl;
        std::cout << "Setpoint: " << setpoint << " V" << std::endl;
        std::cout << "Time Step: " << timeStep << " s" << std::endl;
        std::cout << "Circuit Time Constant: " << (10.0 * 0.05) << " s" << std::endl;
        
        // Run simulation and get results
        auto results = simulator.runSimulation(setpoint, duration, timeStep, printDataPoint);
        
        // Save results to CSV
        saveResultsToCSV(results, "step_response_fixed_completely.csv");
        
        // Analyze performance
        analyzePIDPerformance(results, setpoint);
        
        // ===== Disturbance Rejection Test =====
        std::cout << "\nRunning disturbance rejection test..." << std::endl;
        
        // Set up continuous simulation with callback
        simulator.startContinuousSimulation(setpoint, timeStep, printDataPoint);
        
        // Wait for 5 seconds to let the system stabilize
        std::cout << "System stabilizing..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(duration/3)));
        
        // Add a disturbance
        double disturbanceMagnitude = -2.0;
        std::cout << "Adding disturbance of " << disturbanceMagnitude << " V..." << std::endl;
        simulator.addDisturbance(disturbanceMagnitude);
        
        // Wait for 5 more seconds to observe disturbance rejection
        std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(duration/3)));
        
        // ===== Setpoint Tracking Test =====
        // Change setpoint
        double newSetpoint = 7.0;
        std::cout << "Changing setpoint to " << newSetpoint << " V..." << std::endl;
        simulator.changeSetpoint(newSetpoint);
        
        // Wait for 5 more seconds to observe setpoint tracking
        std::this_thread::sleep_for(std::chrono::seconds(static_cast<int>(duration/3)));
        
        // Stop simulation
        simulator.stopContinuousSimulation();
        
        std::cout << "\nSimulation completed successfully." << std::endl;
        std::cout << "Check 'step_response_fixed_completely.csv' for detailed data." << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}