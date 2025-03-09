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
#include <limits>

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
        
        // Prevent division by zero in case of very small time delta
        if (timeDelta < 1e-10) {
            timeDelta = 1e-10;
        }
        
        // Calculate proportional term - proportional to current error
        proportionalTerm_ = kp_ * error;
        
        // Calculate integral term only if time has passed
        if (timeDelta > 0.0) {
            // Only accumulate integral if not in anti-windup state
            if (!antiWindupActive_) {
                // Accumulate integral sum (error × time)
                // Limit the maximum integral sum to avoid excessive buildup
                double maxIntegralSum = (outputMax_ - outputMin_) / ki_ * 1.5;
                integralSum_ += error * timeDelta;
                // Clamp the integral sum
                integralSum_ = std::clamp(integralSum_, -maxIntegralSum, maxIntegralSum);
            }
            integralTerm_ = ki_ * integralSum_;
        }
        
        // Calculate derivative term only if time has passed
        if (timeDelta > 0.0) {
            // Use filtered derivative to reduce noise sensitivity
            // Rate of change of error over time
            // Limit maximum derivative to avoid spikes
            double rawDerivative = (error - lastError_) / timeDelta;
            const double MAX_DERIVATIVE = 1000.0;
            rawDerivative = std::clamp(rawDerivative, -MAX_DERIVATIVE, MAX_DERIVATIVE);
            derivativeTerm_ = kd_ * rawDerivative;
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
     * @brief Virtual destructor for proper inheritance
     */
    virtual ~ElectricCircuit() = default;
    
    /**
     * @brief Apply an input voltage to the circuit
     * 
     * @param voltage Input voltage in volts
     * @param dt Time step in seconds
     * @throws std::invalid_argument if dt is negative or zero
     */
    virtual void applyVoltage(double voltage, double dt) {
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

protected:
    // Allow derived classes to access the mutex
    // FIX: Made getMutex() const-qualified
    std::mutex& getMutex() const { return mutex_; }
    
    // Allow derived classes to set voltage directly
    void setVoltage(double voltage) {
        // Check for NaN and infinity
        if (std::isnan(voltage) || std::isinf(voltage)) {
            std::cerr << "Warning: Invalid voltage value detected, resetting to 0.0" << std::endl;
            voltage_ = 0.0;
        } else {
            voltage_ = voltage;
        }
        
        current_ = voltage_ / resistance_;
    }
    
    // Expose noise generation to derived classes
    double generateNoise() {
        return noiseLevel_ * noiseDist_(rng_);
    }
    
    // Accessors for protected members
    double getResistance() const { return resistance_; }
    double getCapacitance() const { return capacitance_; }

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
    // FIX: Made mutex mutable to allow modification in const methods
    mutable std::mutex mutex_; // Mutex for thread safety
    
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
        setVoltage(voltage_ + voltageChange);
    }
};

/**
 * @class RLCCircuit
 * @brief A more complex circuit model with resistance, inductance, and capacitance
 * 
 * This class simulates an RLC circuit with voltage, current, resistance,
 * inductance, and capacitance. It exhibits second-order dynamics with
 * potential for resonance and oscillation.
 */
class RLCCircuit : public ElectricCircuit {
public:
    /**
     * @brief Construct a new RLCCircuit object
     * 
     * @param initialVoltage Initial voltage across the capacitor in volts
     * @param initialCurrent Initial current through the inductor in amperes
     * @param resistance Circuit resistance in ohms
     * @param inductance Circuit inductance in henries
     * @param capacitance Circuit capacitance in farads
     * @param noiseLevel Level of noise in the circuit (0.0 to 1.0)
     * @throws std::invalid_argument if parameters are invalid
     */
    RLCCircuit(double initialVoltage, double initialCurrent, double resistance, 
               double inductance, double capacitance, double noiseLevel)
        : ElectricCircuit(initialVoltage, resistance, capacitance, noiseLevel),
          inductance_(inductance), current_(initialCurrent), 
          previousCurrent_(initialCurrent) {
        
        // Validate parameter
        if (inductance <= 0.0) {
            throw std::invalid_argument("Inductance must be positive");
        }
        
        // Calculate resonant frequency and damping ratio
        resonantFrequency_ = 1.0 / (2.0 * M_PI * std::sqrt(inductance * capacitance));
        dampingRatio_ = resistance / (2.0 * std::sqrt(inductance / capacitance));
        
        std::cout << "RLC Circuit characteristics:" << std::endl;
        std::cout << "  Resonant frequency: " << resonantFrequency_ << " Hz" << std::endl;
        std::cout << "  Damping ratio: " << dampingRatio_ << std::endl;
        
        if (dampingRatio_ < 1.0) {
            std::cout << "  Circuit is underdamped - will exhibit oscillation" << std::endl;
        } else if (std::abs(dampingRatio_ - 1.0) < 0.01) {
            std::cout << "  Circuit is critically damped" << std::endl;
        } else {
            std::cout << "  Circuit is overdamped" << std::endl;
        }
    }
    
    /**
     * @brief Get the inductance of the circuit
     * 
     * @return double Inductance in henries
     */
    double getInductance() const {
        return inductance_;
    }
    
    /**
     * @brief Get the resonant frequency of the circuit
     * 
     * @return double Resonant frequency in hertz
     */
    double getResonantFrequency() const {
        return resonantFrequency_;
    }
    
    /**
     * @brief Get the damping ratio of the circuit
     * 
     * @return double Damping ratio (dimensionless)
     */
    double getDampingRatio() const {
        return dampingRatio_;
    }
    
    /**
     * @brief Apply an input voltage to the circuit
     * 
     * @param voltage Input voltage in volts
     * @param dt Time step in seconds
     * @throws std::invalid_argument if dt is negative or zero
     */
    void applyVoltage(double voltage, double dt) override {
        if (dt <= 0.0) {
            throw std::invalid_argument("Time step must be positive");
        }
        
        std::lock_guard<std::mutex> lock(getMutex()); // Lock for thread safety
        
        // Update circuit state with the new voltage
        updateRLCCircuitState(voltage, dt);
    }
    
    /**
     * @brief Get the current through the inductor
     * 
     * @return double Current in amperes
     */
    double getInductorCurrent() const {
        std::lock_guard<std::mutex> lock(getMutex()); // This now works with the const-qualified getMutex()
        return current_;
    }
    
private:
    double inductance_;      // Circuit inductance (H)
    double current_;         // Current through the inductor (A)
    double previousCurrent_; // Previous current for solver (A)
    double resonantFrequency_; // Resonant frequency (Hz)
    double dampingRatio_;    // Damping ratio (dimensionless)
    
    /**
     * @brief Update RLC circuit state based on circuit equations
     * 
     * @param inputVoltage Input voltage
     * @param dt Time step
     */
    void updateRLCCircuitState(double inputVoltage, double dt) {
        // Get current values and make sure they're not too small to cause numerical issues
        double capacitorVoltage = getVoltage();
        double resistance = getResistance();
        double capacitance = getCapacitance();
        
        // Add safety checks to prevent division by zero
        const double MIN_VALUE = 1e-10;
        
        if (inductance_ < MIN_VALUE) {
            std::cerr << "Warning: Inductance too small, using minimum value" << std::endl;
            inductance_ = MIN_VALUE;
        }
        
        // Use a smaller time step for numerical stability
        // Split the calculation into multiple substeps
        const int NUM_SUBSTEPS = 10;
        const double subDt = dt / NUM_SUBSTEPS;
        
        for (int i = 0; i < NUM_SUBSTEPS; i++) {
            // Calculate voltage across the inductor: V_L = L * dI/dt
            // In the RLC circuit: V_in = V_R + V_L + V_C
            // V_R = I*R
            // V_L = L * dI/dt
            // V_C = capacitorVoltage
            
            // Therefore: dI/dt = (V_in - I*R - V_C) / L
            double currentDerivative = (inputVoltage - current_ * resistance - capacitorVoltage) / inductance_;
            
            // Safety bounds check
            const double MAX_DERIVATIVE = 1e6;
            if (std::abs(currentDerivative) > MAX_DERIVATIVE) {
                std::cerr << "Warning: Current derivative too large, limiting" << std::endl;
                currentDerivative = (currentDerivative > 0) ? MAX_DERIVATIVE : -MAX_DERIVATIVE;
            }
            
            // Update current using Euler method with small time step
            double newCurrent = current_ + currentDerivative * subDt;
            
            // Check for NaN/Inf and reset if needed
            if (std::isnan(newCurrent) || std::isinf(newCurrent)) {
                std::cerr << "Warning: Invalid current value, resetting" << std::endl;
                newCurrent = 0.0;
            }
            
            current_ = newCurrent;
            
            // Calculate capacitor voltage change
            double voltageChange = (current_ / capacitance) * subDt;
            
            // Add noise only on first substep, scaled appropriately
            if (i == 0) {
                voltageChange += generateNoise() * 0.1; // Reduced noise for stability
            }
            
            // Update capacitor voltage
            double newVoltage = capacitorVoltage + voltageChange;
            
            // Safety check for capacitor voltage
            if (std::isnan(newVoltage) || std::isinf(newVoltage)) {
                std::cerr << "Warning: Invalid voltage value, resetting" << std::endl;
                newVoltage = 0.0;
            }
            
            setVoltage(newVoltage);
            capacitorVoltage = newVoltage;
            
            // Debug output every few substeps on the first few iterations
            static int debugCounter = 0;
            if (debugCounter < 100 && i == 0) {
                std::cout << "Debug: Substep " << i 
                          << " Current=" << current_
                          << " Voltage=" << capacitorVoltage
                          << " dI/dt=" << currentDerivative << std::endl;
                debugCounter++;
            }
        }
        
        // Store current for next time step
        previousCurrent_ = current_;
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
        
        // Set maximum number of iterations as a safety mechanism
        const size_t MAX_ITERATIONS = 1000000; // Prevent infinite loops
        size_t iterations = 0;
        
        try {
            // Run simulation for specified duration
            double currentTime = 0.0;
            while (currentTime <= duration) {
                // Safety check to prevent infinite loops
                if (++iterations > MAX_ITERATIONS) {
                    std::cerr << "Warning: Maximum iteration count reached, simulation stopped" << std::endl;
                    break;
                }
                
                // Print debug info every 100 iterations
                if (iterations % 100 == 0) {
                    std::cout << "Debug: Iteration " << iterations 
                              << ", simulation time: " << currentTime << "s" << std::endl;
                }
                
                // Run a single simulation step
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
        } catch (const std::exception& e) {
            std::cerr << "Error during simulation: " << e.what() << std::endl;
            // Return whatever results we have so far
        }
        
        std::cout << "Simulation completed with " << results.size() 
                  << " data points over " << iterations << " iterations" << std::endl;
        
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
        try {
            // Get current circuit voltage
            double currentVoltage = circuit_->getVoltage();
            
            // Safety check for NaN or infinity
            if (std::isnan(currentVoltage) || std::isinf(currentVoltage)) {
                std::cerr << "Warning: Invalid voltage detected, resetting to 0.0" << std::endl;
                currentVoltage = 0.0;
            }
            
            // Compute control output using PID controller
            double controlOutput = pidController_->compute(setpoint, currentVoltage, timeStep);
            
            // Apply control output to circuit
            circuit_->applyVoltage(controlOutput, timeStep);
            
            // Get updated voltage after applying control
            double updatedVoltage = circuit_->getVoltage();
            
            // Create data point with current values
            DataPoint dataPoint;
            dataPoint.time = currentTime;
            dataPoint.setpoint = setpoint;
            dataPoint.measured = updatedVoltage;
            dataPoint.control = controlOutput;
            dataPoint.p_term = pidController_->getProportionalTerm();
            dataPoint.i_term = pidController_->getIntegralTerm();
            dataPoint.d_term = pidController_->getDerivativeTerm();
            dataPoint.anti_windup = pidController_->isAntiWindupActive();
            
            return dataPoint;
        } catch (const std::exception& e) {
            std::cerr << "Error in simulation step at time " << currentTime 
                      << "s: " << e.what() << std::endl;
            
            // Return a safe default data point in case of error
            DataPoint errorPoint;
            errorPoint.time = currentTime;
            errorPoint.setpoint = setpoint;
            errorPoint.measured = 0.0;
            errorPoint.control = 0.0;
            errorPoint.p_term = 0.0;
            errorPoint.i_term = 0.0;
            errorPoint.d_term = 0.0;
            errorPoint.anti_windup = false;
            
            return errorPoint;
        }
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
        size_t iterationCount = 0;
        const size_t MAX_ITERATIONS = 1000000; // Safety limit
        
        // Run simulation until stopped
        while (true) {
            // Check for safety limit
            if (++iterationCount > MAX_ITERATIONS) {
                std::cerr << "Warning: Maximum iteration count reached in continuous simulation" << std::endl;
                break;
            }
            
            // Check if simulation should continue
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!isRunning_) {
                    break;  // Exit the loop if simulation is stopped
                }
                localSetpoint = currentSetpoint_;  // Get the latest setpoint
            }
            
            try {
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
            } catch (const std::exception& e) {
                std::cerr << "Error in continuous simulation thread: " << e.what() << std::endl;
                // Continue despite error
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
        double percentComplete = 0.0;
        if (std::abs(targetChange) > 1e-6) { // Avoid division by near-zero
            percentComplete = (results[i].measured - initialValue) / targetChange;
        }
        
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
        
        std::cout << "\n1. RC Circuit Simulation" << std::endl;
        std::cout << "----------------------" << std::endl;
        
        // Create PID controller for RC circuit
        auto pidRC = std::make_shared<PIDController>(0.15, 0.03, 0.01, 0.0, 12.0);
        
        // Create RC circuit
        auto rcCircuit = std::make_shared<ElectricCircuit>(0.0, 10.0, 0.05, 0.005);
        
        // Create circuit simulator
        CircuitSimulator simulatorRC(pidRC, rcCircuit);
        
        // Run RC simulation
        double setpointRC = 5.0;
        double durationRC = 15.0;
        double timeStepRC = 0.01;
        
        std::cout << "Running RC circuit simulation..." << std::endl;
        std::cout << "Setpoint: " << setpointRC << " V" << std::endl;
        std::cout << "Time Step: " << timeStepRC << " s" << std::endl;
        std::cout << "Circuit Time Constant: " << (10.0 * 0.05) << " s" << std::endl;
        
        auto resultsRC = simulatorRC.runSimulation(setpointRC, durationRC, timeStepRC, printDataPoint);
        saveResultsToCSV(resultsRC, "rc_circuit_response.csv");
        analyzePIDPerformance(resultsRC, setpointRC);
        
        // Now run RLC circuit simulation with improved parameters
        std::cout << "\n2. RLC Circuit Simulation" << std::endl;
        std::cout << "------------------------" << std::endl;
        
        // Create RLC circuit with parameters closer to critical damping
        // Changed resistance from 10.0 to 4.47 to get damping ratio closer to 1.0
        auto rlcCircuit = std::make_shared<RLCCircuit>(0.0, 0.0, 4.47, 0.2, 0.05, 0.002);
        
        // Use stronger PID gains for the RLC circuit
        auto pidRLC = std::make_shared<PIDController>(0.8, 0.15, 0.03, 0.0, 12.0);
        
        // Create circuit simulator
        CircuitSimulator simulatorRLC(pidRLC, rlcCircuit);
        
        // Run RLC simulation with smaller time step and longer duration
        double setpointRLC = 5.0;
        double durationRLC = 30.0;  // Longer to ensure settling
        double timeStepRLC = 0.005; // Smaller for numerical stability
        
        std::cout << "Running RLC circuit simulation..." << std::endl;
        std::cout << "Setpoint: " << setpointRLC << " V" << std::endl;
        std::cout << "Time Step: " << timeStepRLC << " s" << std::endl;
        std::cout << "Maximum run time: " << durationRLC << " s" << std::endl;
        
        auto resultsRLC = simulatorRLC.runSimulation(setpointRLC, durationRLC, timeStepRLC, printDataPoint);
        saveResultsToCSV(resultsRLC, "rlc_circuit_response.csv");
        analyzePIDPerformance(resultsRLC, setpointRLC);
        
        // Run a disturbance test on the RLC circuit
        std::cout << "\n3. RLC Circuit Disturbance Test" << std::endl;
        std::cout << "------------------------------" << std::endl;
        
        simulatorRLC.startContinuousSimulation(setpointRLC, timeStepRLC, printDataPoint);
        
        // Wait for system to stabilize
        std::cout << "System stabilizing..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        // Add a disturbance
        double disturbanceMagnitude = -2.0;
        std::cout << "Adding disturbance of " << disturbanceMagnitude << " V..." << std::endl;
        simulatorRLC.addDisturbance(disturbanceMagnitude);
        
        // Wait to observe disturbance rejection
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        // Change setpoint
        double newSetpoint = 7.0;
        std::cout << "Changing setpoint to " << newSetpoint << " V..." << std::endl;
        simulatorRLC.changeSetpoint(newSetpoint);
        
        // Wait to observe setpoint tracking
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        // Stop simulation
        simulatorRLC.stopContinuousSimulation();
        
        // Compare the results
        std::cout << "\nComparison of RC vs RLC Circuit Control Performance:" << std::endl;
        std::cout << "---------------------------------------------------" << std::endl;
        std::cout << "Note the differences in rise time, overshoot, and settling behavior." << std::endl;
        std::cout << "The RLC circuit exhibits more complex dynamics due to its second-order nature." << std::endl;
        std::cout << "The modified RLC circuit with critical damping and tuned PID parameters" << std::endl;
        std::cout << "should now reach and maintain the setpoint without excessive oscillation." << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}