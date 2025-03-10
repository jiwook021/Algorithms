#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <chrono>
#include <limits>
#include <fstream>
#include <string>
#include <thread>
#include <atomic>
#include <iomanip> // For std::setprecision
#include <signal.h> // For signal handling
#include <unistd.h> // For sleep

// Global flag for graceful shutdown
std::atomic<bool> g_running(true);

// Signal handler to catch termination signals
void signal_handler(int sig) {
    std::cerr << "\nReceived signal " << sig << std::endl;
    g_running.store(false);
    
    // Map signal to name for better debugging
    std::string signal_name;
    switch (sig) {
        case SIGSEGV: signal_name = "Segmentation fault (SIGSEGV)"; break;
        case SIGABRT: signal_name = "Abort (SIGABRT)"; break;
        case SIGFPE: signal_name = "Floating-point exception (SIGFPE)"; break;
        case SIGTERM: signal_name = "Termination (SIGTERM)"; break;
        case SIGINT: signal_name = "Interrupt (SIGINT)"; break;
        default: signal_name = "Unknown"; break;
    }
    
    std::cerr << "Signal type: " << signal_name << std::endl;
    std::cerr << "The program will attempt to exit gracefully." << std::endl;
    
    // Give some time for cleanup - this might prevent immediate termination
    sleep(1);
}

// Forward declarations
class Environment;

/**
 * @brief Traditional PID Controller class
 * 
 * Implements a standard PID controller with configurable gains.
 * Thread-safe implementation with mutex protection for parameter updates.
 */
class PIDController {
public:
    PIDController(double kp = 1.0, double ki = 0.0, double kd = 0.0, double dt = 0.01)
        : kp_(kp), ki_(ki), kd_(kd), dt_(dt), previous_error_(0.0), integral_(0.0) {
    }

    /**
     * @brief Updates the PID parameters
     * 
     * Thread-safe method to update the PID gains.
     * 
     * @param kp Proportional gain
     * @param ki Integral gain
     * @param kd Derivative gain
     */
    void updateParameters(double kp, double ki, double kd) {
        // Lock needed to prevent concurrent access during parameter updates
        std::lock_guard<std::mutex> lock(mutex_);
        kp_ = kp;
        ki_ = ki;
        kd_ = kd;
    }

    /**
     * @brief Computes the control signal based on the error
     * 
     * Thread-safe PID calculation.
     * 
     * @param error Current error (setpoint - measured_value)
     * @return double Control signal
     */
    double compute(double error) {
        // Lock needed to ensure consistent parameter usage during computation
        std::lock_guard<std::mutex> lock(mutex_);

        // Proportional term
        double p_term = kp_ * error;

        // Integral term - with anti-windup
        integral_ += error * dt_;
        // Anti-windup - limit the integral term
        const double MAX_INTEGRAL = 10.0 / (ki_ > 0.01 ? ki_ : 0.01);  // Ensure max contribution is reasonable
        integral_ = std::max(-MAX_INTEGRAL, std::min(MAX_INTEGRAL, integral_));
        double i_term = ki_ * integral_;

        // Derivative term with filtering
        // Use a simple first-order filter to reduce noise sensitivity
        double derivative = (error - previous_error_) / dt_;
        // Simple low-pass filter
        const double FILTER_COEFF = 0.1;
        static double filtered_derivative = 0;
        filtered_derivative = FILTER_COEFF * derivative + (1 - FILTER_COEFF) * filtered_derivative;
        double d_term = kd_ * filtered_derivative;

        // Update previous error
        previous_error_ = error;

        // PID output with clipping to prevent extreme values
        double output = p_term + i_term + d_term;
        
        // Clip the control signal to prevent numerical instability
        const double MAX_CONTROL = 10.0;
        return std::max(-MAX_CONTROL, std::min(MAX_CONTROL, output));
    }

    /**
     * @brief Resets the controller's internal state
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        previous_error_ = 0.0;
        integral_ = 0.0;
    }

    /**
     * @brief Get current PID parameters
     * 
     * @return std::vector<double> Vector containing [kp, ki, kd]
     */
    std::vector<double> getParameters() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {kp_, ki_, kd_};
    }

private:
    double kp_;             // Proportional gain
    double ki_;             // Integral gain
    double kd_;             // Derivative gain
    double dt_;             // Time step
    double previous_error_; // Previous error value
    double integral_;       // Accumulated error (integral term)
    
    // Mutex for thread safety
    mutable std::mutex mutex_;
};

/**
 * @brief Neural Network implementation
 * 
 * A simple feed-forward neural network with one hidden layer.
 * Used for Q-function approximation in reinforcement learning.
 */
class NeuralNetwork {
public:
    /**
     * @brief Construct a new Neural Network
     * 
     * @param input_size Number of input neurons
     * @param hidden_size Number of hidden neurons
     * @param output_size Number of output neurons
     * @param learning_rate Learning rate for training
     */
    NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size, double learning_rate = 0.001)
        : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size), 
          learning_rate_(learning_rate) {
        
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 0.1);

        // Initialize weights and biases with random values
        // Xavier/Glorot initialization for better convergence
        double weight_init_factor = std::sqrt(2.0 / (input_size_ + hidden_size_));
        double weight_init_factor2 = std::sqrt(2.0 / (hidden_size_ + output_size_));

        // Input to hidden weights
        w1_.resize(input_size_, std::vector<double>(hidden_size_));
        for (size_t i = 0; i < input_size_; ++i) {
            for (size_t j = 0; j < hidden_size_; ++j) {
                w1_[i][j] = dist(gen) * weight_init_factor;
            }
        }

        // Hidden to output weights
        w2_.resize(hidden_size_, std::vector<double>(output_size_));
        for (size_t i = 0; i < hidden_size_; ++i) {
            for (size_t j = 0; j < output_size_; ++j) {
                w2_[i][j] = dist(gen) * weight_init_factor2;
            }
        }

        // Hidden layer bias
        b1_.resize(hidden_size_, 0.0);
        // Output layer bias
        b2_.resize(output_size_, 0.0);
        
        std::cout << "Neural network created with sizes: " 
                  << input_size_ << "->" << hidden_size_ << "->" << output_size_ 
                  << " and learning rate: " << learning_rate_ << std::endl;
    }

    /**
     * @brief Forward pass through the network
     * 
     * @param input Input values
     * @return std::vector<double> Output values
     */
    std::vector<double> forward(const std::vector<double>& input) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        try {
            std::cout << "Starting forward pass..." << std::endl;
            
            // Input validation
            if (input.size() != input_size_) {
                throw std::invalid_argument("Input size does not match network input size: expected " + 
                                           std::to_string(input_size_) + ", got " + 
                                           std::to_string(input.size()));
            }

            // Check for NaN or Inf in input
            for (size_t i = 0; i < input.size(); ++i) {
                if (std::isnan(input[i]) || std::isinf(input[i])) {
                    throw std::invalid_argument("Input contains NaN or Inf at index " + std::to_string(i));
                }
            }

            // Scale inputs to prevent extreme values
            std::vector<double> scaled_input = input;
            for (size_t i = 0; i < input_size_; ++i) {
                // Clip inputs to a reasonable range
                const double MAX_INPUT = 10.0;
                scaled_input[i] = std::max(-MAX_INPUT, std::min(MAX_INPUT, input[i]));
            }

            std::cout << "Calculating hidden layer..." << std::endl;
            
            // Calculate hidden layer values
            hidden_.resize(hidden_size_);
            for (size_t j = 0; j < hidden_size_; ++j) {
                hidden_[j] = b1_[j];
                for (size_t i = 0; i < input_size_; ++i) {
                    hidden_[j] += scaled_input[i] * w1_[i][j];
                }
                // ReLU activation function
                hidden_[j] = std::max(0.0, hidden_[j]);
            }

            std::cout << "Calculating output layer..." << std::endl;
            
            // Calculate output layer values
            std::vector<double> output(output_size_);
            for (size_t k = 0; k < output_size_; ++k) {
                output[k] = b2_[k];
                for (size_t j = 0; j < hidden_size_; ++j) {
                    output[k] += hidden_[j] * w2_[j][k];
                }
                // Ensure outputs are in the valid range
                output[k] = std::max(-1.0, std::min(1.0, output[k]));
            }

            std::cout << "Forward pass completed successfully" << std::endl;
            return output;
            
        } catch (const std::exception& e) {
            std::cerr << "Error in forward pass: " << e.what() << std::endl;
            // Return a default output (all zeros) in case of error
            return std::vector<double>(output_size_, 0.0);
        }
    }

    /**
     * @brief Train the network using backpropagation
     * 
     * @param input Input values
     * @param target Target output values
     * @return double Loss value (mean squared error)
     */
    double train(const std::vector<double>& input, const std::vector<double>& target) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        double loss = 0.0;
        
        try {
            std::cout << "Starting network training..." << std::endl;
            
            // Input validation
            if (input.size() != input_size_) {
                throw std::invalid_argument("Input size does not match network input size: expected " + 
                                           std::to_string(input_size_) + ", got " + 
                                           std::to_string(input.size()));
            }
            if (target.size() != output_size_) {
                throw std::invalid_argument("Target size does not match network output size: expected " + 
                                           std::to_string(output_size_) + ", got " + 
                                           std::to_string(target.size()));
            }

            // Check for NaN or Inf in input and target
            for (size_t i = 0; i < input.size(); ++i) {
                if (std::isnan(input[i]) || std::isinf(input[i])) {
                    throw std::invalid_argument("Input contains NaN or Inf at index " + std::to_string(i));
                }
            }
            for (size_t i = 0; i < target.size(); ++i) {
                if (std::isnan(target[i]) || std::isinf(target[i])) {
                    throw std::invalid_argument("Target contains NaN or Inf at index " + std::to_string(i));
                }
            }

            // Scale inputs to prevent extreme values
            std::vector<double> scaled_input = input;
            for (size_t i = 0; i < input_size_; ++i) {
                // Clip inputs to a reasonable range
                const double MAX_INPUT = 10.0;
                scaled_input[i] = std::max(-MAX_INPUT, std::min(MAX_INPUT, input[i]));
            }

            std::cout << "Performing forward pass for training..." << std::endl;
            
            // Forward pass (use a separate vector to avoid modifying the class member)
            std::vector<double> hidden_vals(hidden_size_);
            for (size_t j = 0; j < hidden_size_; ++j) {
                hidden_vals[j] = b1_[j];
                for (size_t i = 0; i < input_size_; ++i) {
                    hidden_vals[j] += scaled_input[i] * w1_[i][j];
                }
                // ReLU activation function
                hidden_vals[j] = std::max(0.0, hidden_vals[j]);
            }
            
            std::vector<double> output(output_size_);
            for (size_t k = 0; k < output_size_; ++k) {
                output[k] = b2_[k];
                for (size_t j = 0; j < hidden_size_; ++j) {
                    output[k] += hidden_vals[j] * w2_[j][k];
                }
                // Ensure outputs are in valid range
                output[k] = std::max(-1.0, std::min(1.0, output[k]));
            }
            
            // Calculate loss (mean squared error)
            loss = 0.0;
            for (size_t k = 0; k < output_size_; ++k) {
                double error = target[k] - output[k];
                loss += error * error;
            }
            loss /= output_size_;
            
            std::cout << "Calculating output layer gradients..." << std::endl;
            
            // Backpropagation - Output layer gradients
            std::vector<double> output_delta(output_size_);
            for (size_t k = 0; k < output_size_; ++k) {
                output_delta[k] = (target[k] - output[k]) * 2.0 / output_size_;
                
                // Clip gradients to prevent explosion
                const double MAX_GRAD = 1.0;
                output_delta[k] = std::max(-MAX_GRAD, std::min(MAX_GRAD, output_delta[k]));
            }
            
            std::cout << "Updating hidden->output weights..." << std::endl;

            // Update hidden to output weights and biases
            for (size_t j = 0; j < hidden_size_; ++j) {
                for (size_t k = 0; k < output_size_; ++k) {
                    double delta_w = learning_rate_ * hidden_vals[j] * output_delta[k];
                    // Clip weight updates
                    const double MAX_UPDATE = 0.1;
                    delta_w = std::max(-MAX_UPDATE, std::min(MAX_UPDATE, delta_w));
                    w2_[j][k] += delta_w;
                }
            }
            for (size_t k = 0; k < output_size_; ++k) {
                double delta_b = learning_rate_ * output_delta[k];
                // Clip bias updates
                const double MAX_UPDATE = 0.1;
                delta_b = std::max(-MAX_UPDATE, std::min(MAX_UPDATE, delta_b));
                b2_[k] += delta_b;
            }
            
            std::cout << "Calculating hidden layer gradients..." << std::endl;

            // Backpropagation - Hidden layer gradients
            std::vector<double> hidden_delta(hidden_size_, 0.0);
            for (size_t j = 0; j < hidden_size_; ++j) {
                for (size_t k = 0; k < output_size_; ++k) {
                    hidden_delta[j] += output_delta[k] * w2_[j][k];
                }
                // ReLU derivative
                hidden_delta[j] *= (hidden_vals[j] > 0.0) ? 1.0 : 0.0;
                
                // Clip gradients
                const double MAX_GRAD = 1.0;
                hidden_delta[j] = std::max(-MAX_GRAD, std::min(MAX_GRAD, hidden_delta[j]));
            }
            
            std::cout << "Updating input->hidden weights..." << std::endl;

            // Update input to hidden weights and biases
            for (size_t i = 0; i < input_size_; ++i) {
                for (size_t j = 0; j < hidden_size_; ++j) {
                    double delta_w = learning_rate_ * scaled_input[i] * hidden_delta[j];
                    // Clip weight updates
                    const double MAX_UPDATE = 0.1;
                    delta_w = std::max(-MAX_UPDATE, std::min(MAX_UPDATE, delta_w));
                    w1_[i][j] += delta_w;
                }
            }
            for (size_t j = 0; j < hidden_size_; ++j) {
                double delta_b = learning_rate_ * hidden_delta[j];
                // Clip bias updates
                const double MAX_UPDATE = 0.1;
                delta_b = std::max(-MAX_UPDATE, std::min(MAX_UPDATE, delta_b));
                b1_[j] += delta_b;
            }
            
            std::cout << "Training completed with loss: " << loss << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error in neural network training: " << e.what() << std::endl;
        }

        return loss;
    }

    /**
     * @brief Save the neural network model to a file
     * 
     * @param filename File to save to
     */
    void saveModel(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            std::ofstream file(filename);
            
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for saving model: " + filename);
            }

            // Save network architecture
            file << input_size_ << " " << hidden_size_ << " " << output_size_ << " " << learning_rate_ << std::endl;
            
            // Save weights w1
            for (size_t i = 0; i < input_size_; ++i) {
                for (size_t j = 0; j < hidden_size_; ++j) {
                    file << w1_[i][j] << " ";
                }
                file << std::endl;
            }
            
            // Save weights w2
            for (size_t i = 0; i < hidden_size_; ++i) {
                for (size_t j = 0; j < output_size_; ++j) {
                    file << w2_[i][j] << " ";
                }
                file << std::endl;
            }
            
            // Save biases b1
            for (size_t i = 0; i < hidden_size_; ++i) {
                file << b1_[i] << " ";
            }
            file << std::endl;
            
            // Save biases b2
            for (size_t i = 0; i < output_size_; ++i) {
                file << b2_[i] << " ";
            }
            
            file.close();
            std::cout << "Model successfully saved to: " << filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error saving model: " << e.what() << std::endl;
        }
    }

    /**
     * @brief Load a neural network model from a file
     * 
     * @param filename File to load from
     */
    void loadModel(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            std::ifstream file(filename);
            
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file for loading model: " + filename);
            }

            // Load network architecture
            file >> input_size_ >> hidden_size_ >> output_size_ >> learning_rate_;
            
            // Resize vectors
            w1_.resize(input_size_, std::vector<double>(hidden_size_));
            w2_.resize(hidden_size_, std::vector<double>(output_size_));
            b1_.resize(hidden_size_);
            b2_.resize(output_size_);
            hidden_.resize(hidden_size_);
            
            // Load weights w1
            for (size_t i = 0; i < input_size_; ++i) {
                for (size_t j = 0; j < hidden_size_; ++j) {
                    file >> w1_[i][j];
                }
            }
            
            // Load weights w2
            for (size_t i = 0; i < hidden_size_; ++i) {
                for (size_t j = 0; j < output_size_; ++j) {
                    file >> w2_[i][j];
                }
            }
            
            // Load biases b1
            for (size_t i = 0; i < hidden_size_; ++i) {
                file >> b1_[i];
            }
            
            // Load biases b2
            for (size_t i = 0; i < output_size_; ++i) {
                file >> b2_[i];
            }
            
            file.close();
            std::cout << "Model successfully loaded from: " << filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw; // Re-throw to notify caller
        }
    }

private:
    size_t input_size_;     // Number of input neurons
    size_t hidden_size_;    // Number of hidden neurons
    size_t output_size_;    // Number of output neurons
    double learning_rate_;  // Learning rate for training
    
    std::vector<std::vector<double>> w1_; // Weights from input to hidden layer
    std::vector<std::vector<double>> w2_; // Weights from hidden to output layer
    std::vector<double> b1_;              // Biases for hidden layer
    std::vector<double> b2_;              // Biases for output layer
    std::vector<double> hidden_;          // Hidden layer activations (stored for backprop)
    
    // Mutex for thread safety
    mutable std::mutex mutex_;
};

/**
 * @brief Interface for the environment used in reinforcement learning
 */
class Environment {
public:
    virtual ~Environment() = default;
    
    /**
     * @brief Reset the environment to initial state
     * 
     * @return std::vector<double> Initial state
     */
    virtual std::vector<double> reset() = 0;
    
    /**
     * @brief Take a step in the environment based on the action
     * 
     * @param action Action to take (PID parameters)
     * @param control_signal Control signal from the PID controller
     * @return std::tuple<std::vector<double>, double, bool> New state, reward, done flag
     */
    virtual std::tuple<std::vector<double>, double, bool> step(
        const std::vector<double>& action, double control_signal) = 0;
    
    /**
     * @brief Get current setpoint for the controller
     * 
     * @return double Current setpoint
     */
    virtual double getSetpoint() const = 0;
    
    /**
     * @brief Get current process value
     * 
     * @return double Current process value
     */
    virtual double getProcessValue() const = 0;
};

/**
 * @brief Simple process simulation environment for testing the PID controller
 * 
 * Simulates a first-order dynamic system with optional disturbances.
 * Modified to provide better stability and prevent extreme values.
 */
class SimpleProcessEnvironment : public Environment {
public:
    /**
     * @brief Construct a new Simple Process Environment
     * 
     * @param time_constant System time constant
     * @param gain System gain
     * @param dt Time step for simulation
     * @param disturbance_std_dev Standard deviation of random disturbances
     */
    SimpleProcessEnvironment(double time_constant = 1.0, double gain = 1.0, 
                              double dt = 0.01, double disturbance_std_dev = 0.0)
        : time_constant_(time_constant), gain_(gain), dt_(dt),
          disturbance_std_dev_(disturbance_std_dev),
          process_value_(0.0), setpoint_(0.0), step_count_(0),
          max_steps_(1000), gen_(std::random_device{}()),
          dist_(0.0, disturbance_std_dev) {
        std::cout << "Created process environment with:" << std::endl
                  << "- Time constant: " << time_constant_ << std::endl
                  << "- Gain: " << gain_ << std::endl
                  << "- Time step: " << dt_ << std::endl
                  << "- Disturbance std dev: " << disturbance_std_dev_ << std::endl;
    }

    /**
     * @brief Reset the environment to initial state
     * 
     * @return std::vector<double> Initial state (error, error derivative, previous control_signal)
     */
    std::vector<double> reset() override {
        step_count_ = 0;
        
        // Fixed initialization for stability
        process_value_ = 0.0;  // Start at 0
        setpoint_ = 1.0;       // Fixed setpoint of 1.0 (reduced from 2.0 for stability)
        
        std::cout << "Environment reset with:" << std::endl
                  << "- Process value: " << process_value_ << std::endl
                  << "- Setpoint: " << setpoint_ << std::endl;
        
        double error = setpoint_ - process_value_;
        std::cout << "- Initial error: " << error << std::endl;
        
        return {error, 0.0, 0.0}; // Initial error, error derivative, previous control
    }

    /**
     * @brief Take a step in the environment with the given PID parameters
     * 
     * @param action PID parameters [kp, ki, kd]
     * @param control_signal Control signal computed by the PID controller
     * @return std::tuple<std::vector<double>, double, bool> New state, reward, done flag
     */
    std::tuple<std::vector<double>, double, bool> step(
        const std::vector<double>& action, double control_signal) override {
        
        // Clip control signal to prevent extreme values
        const double MAX_CONTROL = 10.0;
        double safe_control = std::max(-MAX_CONTROL, std::min(MAX_CONTROL, control_signal));
        
        // Apply control signal to the process (first-order dynamics)
        double prev_value = process_value_;
        
        // First-order system dynamics with bounds
        double derivative = (gain_ * safe_control - process_value_) / time_constant_;
        
        // Limit the rate of change to prevent numerical issues
        const double MAX_RATE = 1.0;
        derivative = std::max(-MAX_RATE, std::min(MAX_RATE, derivative));
        
        // Euler integration
        process_value_ += dt_ * derivative;
        
        // Add disturbance if enabled (with limits)
        if (disturbance_std_dev_ > 0.0) {
            double disturbance = dist_(gen_);
            const double MAX_DISTURBANCE = 0.1;
            disturbance = std::max(-MAX_DISTURBANCE, std::min(MAX_DISTURBANCE, disturbance));
            process_value_ += disturbance;
        }
        
        // Calculate error and its derivative
        double prev_error = setpoint_ - prev_value;
        double error = setpoint_ - process_value_;
        
        // Limit error rate of change
        double error_derivative = (error - prev_error) / dt_;
        const double MAX_ERROR_RATE = 5.0;
        error_derivative = std::max(-MAX_ERROR_RATE, std::min(MAX_ERROR_RATE, error_derivative));
        
        // State: error, error derivative, previous control signal (all clipped)
        std::vector<double> state = {
            error,
            error_derivative,
            safe_control  // Use clipped control signal in state
        };
        
        // Calculate reward - with clamping to prevent extreme values
        // Use a smoother reward function and scaling for better stability
        double error_reward = -std::min(5.0, std::pow(error, 2));
        double control_reward = -0.01 * std::min(5.0, std::pow(safe_control, 2));
        double reward = error_reward + control_reward;
        
        // Add a small stability bonus for being close to setpoint
        if (std::abs(error) < 0.1) {
            reward += 0.5;
        }
        
        // Check if episode is done - only consider max steps, not error threshold
        step_count_++;
        bool done = step_count_ >= max_steps_;
        
        // Debug output periodically
        if (step_count_ % 100 == 0 || step_count_ == 1) {
            std::cout << "Step " << step_count_ 
                      << ": PV=" << std::fixed << std::setprecision(6) << process_value_ 
                      << ", Error=" << error
                      << ", Control=" << safe_control
                      << ", Reward=" << reward
                      << ", Done=" << done
                      << std::endl;
        }
        
        return {state, reward, done};
    }

    /**
     * @brief Get current setpoint
     * 
     * @return double Current setpoint
     */
    double getSetpoint() const override {
        return setpoint_;
    }

    /**
     * @brief Get current process value
     * 
     * @return double Current process value
     */
    double getProcessValue() const override {
        return process_value_;
    }

    /**
     * @brief Set a new setpoint for the process
     * 
     * @param setpoint New setpoint value
     */
    void setSetpoint(double setpoint) {
        // Limit the setpoint to prevent extreme values
        const double MAX_SETPOINT = 5.0;
        setpoint_ = std::max(-MAX_SETPOINT, std::min(MAX_SETPOINT, setpoint));
    }

    /**
     * @brief Set maximum number of steps per episode
     * 
     * @param max_steps Maximum number of steps
     */
    void setMaxSteps(int max_steps) {
        max_steps_ = max_steps;
    }

private:
    double time_constant_;      // System time constant
    double gain_;               // System gain
    double dt_;                 // Time step
    double disturbance_std_dev_; // Standard deviation of disturbances
    
    double process_value_;      // Current process value
    double setpoint_;           // Current setpoint
    int step_count_;            // Current step count
    int max_steps_;             // Maximum steps per episode
    
    // Random number generation for disturbances
    std::mt19937 gen_;
    std::normal_distribution<double> dist_;
};

/**
 * @brief Q-Learning agent for tuning PID parameters
 * 
 * Uses a neural network to approximate the Q-function.
 */
class RLAgent {
public:
    /**
     * @brief Construct a new RL Agent
     * 
     * @param state_size Size of state vector
     * @param action_size Number of actions (PID parameters)
     * @param hidden_size Size of hidden layer in NN
     * @param learning_rate Learning rate for neural network
     * @param gamma Discount factor for future rewards
     * @param epsilon Initial exploration rate
     * @param epsilon_decay Decay rate for epsilon
     * @param epsilon_min Minimum exploration rate
     */
    RLAgent(size_t state_size, size_t action_size, size_t hidden_size = 24,
            double learning_rate = 0.0001, // Reduced learning rate for stability
            double gamma = 0.99,
            double epsilon = 1.0, double epsilon_decay = 0.995, double epsilon_min = 0.01)
        : state_size_(state_size), action_size_(action_size),
          q_network_(state_size, hidden_size, action_size, learning_rate),
          target_network_(state_size, hidden_size, action_size, learning_rate),
          gamma_(gamma), epsilon_(epsilon), epsilon_decay_(epsilon_decay),
          epsilon_min_(epsilon_min), gen_(std::random_device{}()) {
        
        // Copy initial weights from Q-network to target network
        syncTargetNetwork();
        
        // Initialize action ranges (min, max) for each PID parameter
        // Reduced ranges for better stability
        action_ranges_ = {
            {0.0, 2.0},  // kp range (reduced from 10.0)
            {0.0, 1.0},  // ki range (reduced from 5.0)
            {0.0, 0.5}   // kd range (reduced from 2.0)
        };
        
        std::cout << "Created RL Agent with:" << std::endl
                  << "- State size: " << state_size_ << std::endl
                  << "- Action size: " << action_size_ << std::endl
                  << "- Hidden size: " << hidden_size << std::endl
                  << "- Learning rate: " << learning_rate << std::endl
                  << "- Gamma: " << gamma_ << std::endl
                  << "- Initial epsilon: " << epsilon_ << std::endl
                  << "- Action ranges:" << std::endl
                  << "  - Kp: [" << action_ranges_[0].first << ", " << action_ranges_[0].second << "]" << std::endl
                  << "  - Ki: [" << action_ranges_[1].first << ", " << action_ranges_[1].second << "]" << std::endl
                  << "  - Kd: [" << action_ranges_[2].first << ", " << action_ranges_[2].second << "]" << std::endl;
    }

    /**
     * @brief Select action based on current state
     * 
     * Uses epsilon-greedy policy.
     * 
     * @param state Current state
     * @return std::vector<double> Selected action (PID parameters)
     */
    std::vector<double> selectAction(const std::vector<double>& state) {
        std::vector<double> action(action_size_);
        
        try {
            // Check for NaN or Inf in state
            bool has_invalid = false;
            for (size_t i = 0; i < state.size(); ++i) {
                if (std::isnan(state[i]) || std::isinf(state[i])) {
                    has_invalid = true;
                    std::cout << "Invalid state value detected at index " << i << ": " << state[i] << std::endl;
                    break;
                }
            }
            
            // MODIFICATION: Add debug output for state
            std::cout << "State for action selection: [";
            for (size_t i = 0; i < state.size(); ++i) {
                std::cout << std::fixed << std::setprecision(6) << state[i];
                if (i < state.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            // If state has invalid values or epsilon is high, use exploration
            std::uniform_real_distribution<double> eps_dist(0.0, 1.0);
            bool is_exploring = has_invalid || eps_dist(gen_) < epsilon_;
            
            if (is_exploring) {
                // Exploration: random action with constraints
                for (size_t i = 0; i < action_size_; ++i) {
                    std::uniform_real_distribution<double> action_dist(
                        action_ranges_[i].first, action_ranges_[i].second);
                    action[i] = action_dist(gen_);
                }
                
                // Make sure Kp > Ki > Kd for typical stable PID behavior
                if (action_size_ >= 3) {
                    // Ensure Kp is the largest
                    action[0] = std::max(action[0], std::max(action[1], action[2]));
                    // Ensure Kd is the smallest (good for initial stability)
                    action[2] = std::min(action[2], std::min(action[0], action[1]));
                }
                
                std::cout << "Exploring: Random action selected" << std::endl;
            } else {
                try {
                    // Exploitation: best action according to Q-function
                    std::vector<double> q_values = q_network_.forward(state);
                    
                    std::cout << "Q-values: [";
                    for (size_t i = 0; i < q_values.size(); ++i) {
                        std::cout << std::fixed << std::setprecision(6) << q_values[i];
                        if (i < q_values.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                    
                    // For each PID parameter, select the corresponding Q-value
                    for (size_t i = 0; i < action_size_; ++i) {
                        // Scale the Q-value to the action range
                        double normalized_q = (q_values[i] + 1.0) / 2.0;  // Map from [-1,1] to [0,1]
                        double range = action_ranges_[i].second - action_ranges_[i].first;
                        action[i] = action_ranges_[i].first + normalized_q * range;
                    }
                    std::cout << "Exploiting: Action based on Q-values" << std::endl;
                } catch (const std::exception& e) {
                    // Fallback to exploration if exploitation fails
                    std::cerr << "Error during Q-value computation: " << e.what() << std::endl;
                    for (size_t i = 0; i < action_size_; ++i) {
                        std::uniform_real_distribution<double> action_dist(
                            action_ranges_[i].first, action_ranges_[i].second);
                        action[i] = action_dist(gen_);
                    }
                    std::cout << "Exploring (fallback): Random action selected" << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception in selectAction: " << e.what() << std::endl;
            // Provide a safe default action in case of error
            action = {0.5, 0.1, 0.05}; // Conservative PID parameters
            std::cout << "Using safe default action due to error" << std::endl;
        }
        
        std::cout << "Selected action (PID params): [";
        for (size_t i = 0; i < action.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << action[i];
            if (i < action.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        return action;
    }

    /**
     * @brief Update Q-function based on experience
     * 
     * @param state Current state
     * @param action Current action
     * @param reward Reward received
     * @param next_state Next state
     * @param done Whether episode is done
     * @return double Loss from training
     */
    double learn(const std::vector<double>& state, const std::vector<double>& action,
                double reward, const std::vector<double>& next_state, bool done) {
        
        double loss = 0.0;
        
        try {
            std::cout << "Starting learn function..." << std::endl;
            
            // Skip learning for the first few steps (for debugging)
            static int learn_count = 0;
            learn_count++;
            if (learn_count <= 1) {
                std::cout << "Skipping learning for step " << learn_count << " (debugging mode)" << std::endl;
                return 0.0;
            }
            
            // Check for NaN or Inf values in state/action/reward/next_state
            for (size_t i = 0; i < state.size(); ++i) {
                if (std::isnan(state[i]) || std::isinf(state[i])) {
                    std::cout << "WARNING: Invalid state value detected at index " << i << std::endl;
                    return 0.0;  // Skip learning for invalid inputs
                }
            }
            for (size_t i = 0; i < action.size(); ++i) {
                if (std::isnan(action[i]) || std::isinf(action[i])) {
                    std::cout << "WARNING: Invalid action value detected at index " << i << std::endl;
                    return 0.0;  // Skip learning for invalid inputs
                }
            }
            for (size_t i = 0; i < next_state.size(); ++i) {
                if (std::isnan(next_state[i]) || std::isinf(next_state[i])) {
                    std::cout << "WARNING: Invalid next_state value detected at index " << i << std::endl;
                    return 0.0;  // Skip learning for invalid inputs
                }
            }
            if (std::isnan(reward) || std::isinf(reward)) {
                std::cout << "WARNING: Invalid reward value detected" << std::endl;
                return 0.0;  // Skip learning for invalid inputs
            }
            
            std::cout << "Calculating target Q-values..." << std::endl;
            
            // Calculate target Q-value using Bellman equation
            std::vector<double> next_q_values = target_network_.forward(next_state);
            
            std::cout << "Getting current Q-values..." << std::endl;
            
            // Current Q-values for state
            std::vector<double> q_values = q_network_.forward(state);
            std::vector<double> target_q_values = q_values;
            
            std::cout << "Computing normalized actions..." << std::endl;
            
            // Convert PID action back to normalized Q-values for training
            std::vector<double> norm_action(action_size_);
            for (size_t i = 0; i < action_size_; ++i) {
                double range = action_ranges_[i].second - action_ranges_[i].first;
                norm_action[i] = 2.0 * (action[i] - action_ranges_[i].first) / range - 1.0;
            }
            
            std::cout << "Updating target Q-values..." << std::endl;
            
            // Update target for each action component separately
            for (size_t i = 0; i < action_size_; ++i) {
                // Q-learning update
                if (done) {
                    target_q_values[i] = reward;
                } else {
                    target_q_values[i] = reward + gamma_ * next_q_values[i];
                }
                
                // Scale target to match output range of neural network
                target_q_values[i] = std::max(-1.0, std::min(1.0, target_q_values[i]));
            }
            
            std::cout << "Training the Q-network..." << std::endl;
            
            // Train the Q-network
            loss = q_network_.train(state, target_q_values);
            
            // Decay exploration rate
            epsilon_ = std::max(epsilon_min_, epsilon_ * epsilon_decay_);
            
            std::cout << "Learn function completed with loss: " << loss << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Exception during learning: " << e.what() << std::endl;
            // Don't rethrow - we want to continue training
        }
        
        return loss;
    }

    /**
     * @brief Synchronize target network with Q-network
     */
    void syncTargetNetwork() {
        try {
            // Save Q-network to a temporary file
            std::string temp_file = "temp_q_network.dat";
            q_network_.saveModel(temp_file);
            
            // Load the weights into target network
            target_network_.loadModel(temp_file);
            
            // Clean up temporary file
            std::remove(temp_file.c_str());
            
            std::cout << "Target network synchronized with Q-network" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error synchronizing target network: " << e.what() << std::endl;
        }
    }

    /**
     * @brief Save the agent model to a file
     * 
     * @param filename File to save to
     */
    void saveModel(const std::string& filename) {
        try {
            q_network_.saveModel(filename);
            std::cout << "Model saved to: " << filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error saving model: " << e.what() << std::endl;
        }
    }

    /**
     * @brief Load an agent model from a file
     * 
     * @param filename File to load from
     */
    void loadModel(const std::string& filename) {
        try {
            q_network_.loadModel(filename);
            target_network_.loadModel(filename);
            std::cout << "Model loaded from: " << filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw; // Re-throw to notify caller
        }
    }

    /**
     * @brief Get current exploration rate (epsilon)
     */
    double getEpsilon() const {
        return epsilon_;
    }

    /**
     * @brief Simple grid search for PID parameters (alternative to RL)
     * 
     * @param env Environment to test in
     * @param num_tests Number of steps to test each parameter set
     * @return std::vector<double> Best PID parameters found [kp, ki, kd]
     */
    std::vector<double> gridSearch(std::shared_ptr<Environment> env, int num_tests = 100) {
        // Define parameter ranges to test
        std::vector<double> kp_values = {0.2, 0.5, 1.0, 1.5};
        std::vector<double> ki_values = {0.0, 0.1, 0.2, 0.5};
        std::vector<double> kd_values = {0.0, 0.02, 0.05, 0.1};
        
        double best_reward = -std::numeric_limits<double>::infinity();
        std::vector<double> best_params = {0.5, 0.1, 0.05};  // Default reasonable values
        
        std::cout << "Starting PID parameter grid search..." << std::endl;
        
        // Shared PID controller for testing
        auto pid = std::make_shared<PIDController>();
        
        // Test each combination
        for (double kp : kp_values) {
            for (double ki : ki_values) {
                for (double kd : kd_values) {
                    pid->updateParameters(kp, ki, kd);
                    double total_reward = 0.0;
                    
                    // Reset environment
                    env->reset();
                    pid->reset();
                    
                    // Test over multiple steps
                    for (int step = 0; step < num_tests; ++step) {
                        double error = env->getSetpoint() - env->getProcessValue();
                        double control_signal = pid->compute(error);
                        
                        // Take a step
                        std::vector<double> action = {kp, ki, kd};
                        auto [_, reward, done] = env->step(action, control_signal);
                        
                        total_reward += reward;
                        
                        if (done) break;
                    }
                    
                    // Check if this is the best combination so far
                    if (total_reward > best_reward) {
                        best_reward = total_reward;
                        best_params = {kp, ki, kd};
                        std::cout << "New best params: [" << kp << ", " << ki << ", " << kd 
                                  << "] with total reward: " << total_reward << std::endl;
                    }
                }
            }
        }
        
        std::cout << "Grid search complete. Best parameters: ["
                  << best_params[0] << ", " << best_params[1] << ", " << best_params[2]
                  << "] with reward: " << best_reward << std::endl;
        
        return best_params;
    }

private:
    size_t state_size_;     // Size of state vector
    size_t action_size_;    // Number of actions (PID parameters)
    
    NeuralNetwork q_network_;      // Q-function network
    NeuralNetwork target_network_; // Target network for stable learning
    
    double gamma_;         // Discount factor
    double epsilon_;       // Exploration rate
    double epsilon_decay_; // Decay rate for epsilon
    double epsilon_min_;   // Minimum exploration rate
    
    // Random number generator
    std::mt19937 gen_;
    
    // Action ranges for each PID parameter (min, max)
    std::vector<std::pair<double, double>> action_ranges_;
};

/**
 * @brief Self-tuning PID Controller system that uses RL for parameter optimization
 */
class SelfTuningPIDController {
public:
    /**
     * @brief Construct a new Self Tuning PID Controller
     * 
     * @param env Environment to control
     * @param pid_controller PID controller
     * @param hidden_size Hidden layer size for RL agent's neural network
     */
    SelfTuningPIDController(std::shared_ptr<Environment> env,
                            std::shared_ptr<PIDController> pid_controller,
                            size_t hidden_size = 24)
        : env_(env), pid_controller_(pid_controller),
          rl_agent_(3, 3, hidden_size), // State size: 3, Action size: 3 (kp, ki, kd)
          is_training_(false) {
        std::cout << "Created Self-tuning PID Controller" << std::endl;
    }

    /**
     * @brief Start the training process
     * 
     * @param num_episodes Number of training episodes
     * @param target_sync_frequency Frequency to sync target network
     * @param print_frequency Frequency to print training progress
     */
    void train(int num_episodes, int target_sync_frequency = 10, int print_frequency = 10) {
        // Set training flag
        is_training_.store(true);
        
        double total_reward = 0.0;
        double best_episode_reward = -std::numeric_limits<double>::infinity();
        std::vector<double> best_parameters;
        
        // Training statistics
        int episodes_without_improvement = 0;
        const int max_episodes_without_improvement = 100;  // Early stopping criteria
        
        std::cout << "Starting training for " << num_episodes << " episodes" << std::endl;
        std::cout << "Press Ctrl+C to stop training (signal handling not implemented yet)" << std::endl;
        
        for (int episode = 0; episode < num_episodes && g_running.load(); ++episode) {
            // Check if training was externally stopped
            if (!is_training_.load()) {
                std::cout << "Training was externally stopped at episode " << episode << std::endl;
                break;
            }
            
            std::cout << "\n========== Starting Episode " << episode + 1 << " ==========\n" << std::endl;
            
            // Reset environment and get initial state
            std::vector<double> state = env_->reset();
            pid_controller_->reset();
            
            // Debug the initial state more clearly
            std::cout << "Initial state vector: [";
            for (size_t i = 0; i < state.size(); ++i) {
                std::cout << std::fixed << std::setprecision(6) << state[i];
                if (i < state.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            double episode_reward = 0.0;
            bool done = false;
            int step_count = 0;
            
            // Set a reasonable max step limit
            int max_steps = 100;  // Reduced from 1000 to make episodes shorter for debugging
            
            // Set a higher max step limit for the environment to prevent early termination
            if (auto* simple_env = dynamic_cast<SimpleProcessEnvironment*>(env_.get())) {
                simple_env->setMaxSteps(max_steps);
            }
            
            // Enforce minimum steps to ensure proper training
            int min_steps = 5;  // At least run 5 steps regardless of done flag
            
            while ((step_count < min_steps || (!done && step_count < max_steps)) && 
                    is_training_.load() && g_running.load()) {
                
                try {
                    std::cout << "\n----- Step " << step_count + 1 << " -----" << std::endl;
                    
                    // Select action based on current state
                    std::vector<double> action = rl_agent_.selectAction(state);
                    
                    // Update PID parameters
                    pid_controller_->updateParameters(action[0], action[1], action[2]);
                    auto params = pid_controller_->getParameters();
                    std::cout << "PID parameters set to: ["
                              << std::fixed << std::setprecision(6) << params[0] << ", " 
                              << params[1] << ", " 
                              << params[2] << "]" << std::endl;
                    
                    // Calculate control signal
                    double error = env_->getSetpoint() - env_->getProcessValue();
                    double control_signal = pid_controller_->compute(error);
                    std::cout << "Current error: " << std::fixed << std::setprecision(6) << error
                              << ", Control signal: " << control_signal << std::endl;
                    
                    // Take a step in the environment
                    std::vector<double> next_state;
                    double reward;
                    
                    std::cout << "Taking environment step..." << std::endl;
                    std::tie(next_state, reward, done) = env_->step(action, control_signal);
                    
                    std::cout << "Next state: [";
                    for (size_t i = 0; i < next_state.size(); ++i) {
                        std::cout << std::fixed << std::setprecision(6) << next_state[i];
                        if (i < next_state.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                    std::cout << "Reward: " << std::fixed << std::setprecision(6) << reward 
                              << ", Done: " << done << std::endl;
                    
                    // Train the agent (but skip for the first step for debugging)
                    double loss = 0.0;
                    if (step_count > 0) {
                        std::cout << "Training the agent..." << std::endl;
                        loss = rl_agent_.learn(state, action, reward, next_state, done);
                        std::cout << "Training completed with loss: " << loss << std::endl;
                    } else {
                        std::cout << "Skipping training for first step (debugging)" << std::endl;
                    }
                    
                    // Update state and accumulated reward
                    state = next_state;
                    episode_reward += reward;
                    
                } catch (const std::exception& e) {
                    std::cerr << "Exception during training step: " << e.what() << std::endl;
                    std::cerr << "Continuing with next step..." << std::endl;
                }
                
                step_count++;
                
                // Break from step loop if user requests (after minimum steps)
                if (step_count >= min_steps && step_count % 10 == 0) {
                    std::cout << "Would you like to continue this episode? (y/n): ";
                    char continue_choice;
                    std::cin >> continue_choice;
                    if (continue_choice != 'y' && continue_choice != 'Y') {
                        std::cout << "Breaking from episode after " << step_count << " steps." << std::endl;
                        break;
                    }
                }
            }
            
            // Report why we exited the training loop
            std::cout << "\nEpisode loop exited with: done=" << done 
                      << ", step_count=" << step_count
                      << ", is_training=" << is_training_.load() 
                      << ", g_running=" << g_running.load() << std::endl;
            
            // Check again if training was stopped during episode
            if (!is_training_.load() || !g_running.load()) {
                std::cout << "Training was stopped during episode " << episode << std::endl;
                break;
            }
            
            // Sync target network periodically
            if (episode % target_sync_frequency == 0) {
                rl_agent_.syncTargetNetwork();
            }
            
            // Track improvement
            bool improved = false;
            
            // Save best parameters
            if (episode_reward > best_episode_reward) {
                improved = true;
                episodes_without_improvement = 0;
                best_episode_reward = episode_reward;
                best_parameters = pid_controller_->getParameters();
                
                // Save the best model so far
                std::cout << "New best model found! Saving..." << std::endl;
                rl_agent_.saveModel("best_pid_model.dat");
            } else {
                episodes_without_improvement++;
            }
            
            // Print progress
            total_reward += episode_reward;
            
            std::cout << "\n========== Episode " << (episode + 1) << " Summary ==========" << std::endl;
            std::cout << "Episode Reward: " << std::fixed << std::setprecision(6) << episode_reward << std::endl;
            std::cout << "Steps Completed: " << step_count << std::endl;
            std::cout << "Current Epsilon: " << std::fixed << std::setprecision(6) << rl_agent_.getEpsilon() << std::endl;
            
            if (!best_parameters.empty()) {
                std::cout << "Best Parameters so far: [" 
                          << std::fixed << std::setprecision(6) << best_parameters[0] << ", " 
                          << best_parameters[1] << ", " 
                          << best_parameters[2] << "]" << std::endl;
            }
            
            if (improved) {
                std::cout << "Model improved in this episode!" << std::endl;
            }
            
            std::cout << "Episodes without improvement: " 
                      << episodes_without_improvement << "/" 
                      << max_episodes_without_improvement << std::endl;
            
            // Check if the user wants to continue training
            if (episode < num_episodes - 1) {
                std::cout << "\nContinue training? (y/n): ";
                char continue_choice;
                std::cin >> continue_choice;
                if (continue_choice != 'y' && continue_choice != 'Y') {
                    std::cout << "Training stopped by user after " << (episode + 1) << " episodes." << std::endl;
                    break;
                }
            }
            
            // Early stopping if no improvement for a while
            if (episodes_without_improvement >= max_episodes_without_improvement) {
                std::cout << "Early stopping: No improvement for " 
                          << max_episodes_without_improvement << " episodes" << std::endl;
                break;
            }
        }
        
        // Set the best parameters to the PID controller
        if (!best_parameters.empty()) {
            pid_controller_->updateParameters(
                best_parameters[0], best_parameters[1], best_parameters[2]);
            std::cout << "Training completed. Best PID parameters: ["
                      << std::fixed << std::setprecision(6) << best_parameters[0] << ", "
                      << best_parameters[1] << ", "
                      << best_parameters[2] << "]" << std::endl;
        } else {
            std::cout << "Training completed but no good parameters were found." << std::endl;
        }
        
        // Reset training flag
        is_training_.store(false);
    }

    /**
     * @brief Use grid search to find optimal PID parameters (alternative to RL)
     * 
     * @param num_tests Number of steps to evaluate each parameter set
     */
    void simpleTuning(int num_tests = 100) {
        std::cout << "Starting simple grid search tuning..." << std::endl;
        
        // Use the grid search method from the RL agent
        std::vector<double> best_params = rl_agent_.gridSearch(env_, num_tests);
        
        // Set the best parameters to the PID controller
        pid_controller_->updateParameters(best_params[0], best_params[1], best_params[2]);
        
        std::cout << "Simple tuning completed. Best PID parameters: ["
                  << std::fixed << std::setprecision(6) << best_params[0] << ", "
                  << best_params[1] << ", "
                  << best_params[2] << "]" << std::endl;
    }

    /**
     * @brief Stop the training process
     */
    void stopTraining() {
        is_training_.store(false);
        std::cout << "Training stop requested." << std::endl;
    }

    /**
     * @brief Run the controller with current parameters
     * 
     * @param num_steps Number of steps to run
     * @param setpoint Setpoint value
     * @return std::vector<std::vector<double>> History of [time, setpoint, process_value, control_signal]
     */
    std::vector<std::vector<double>> run(int num_steps, double setpoint) {
        // Vector to store history for plotting
        std::vector<std::vector<double>> history;
        
        std::cout << "Running controller with setpoint " << setpoint 
                  << " for " << num_steps << " steps" << std::endl;
        
        try {
            // Reset environment and controller
            env_->reset();
            pid_controller_->reset();
            
            // Set desired setpoint
            dynamic_cast<SimpleProcessEnvironment*>(env_.get())->setSetpoint(setpoint);
            
            auto pid_params = pid_controller_->getParameters();
            std::cout << "PID parameters: ["
                      << std::fixed << std::setprecision(6) << pid_params[0] << ", " 
                      << pid_params[1] << ", " 
                      << pid_params[2] << "]" << std::endl;
            
            for (int step = 0; step < num_steps && g_running.load(); ++step) {
                // Time in seconds
                double time = step * 0.01;
                
                // Get current state
                double error = env_->getSetpoint() - env_->getProcessValue();
                
                // Calculate control signal
                double control_signal = pid_controller_->compute(error);
                
                // Take a step in the environment
                env_->step(pid_params, control_signal);
                
                // Store history
                history.push_back({time, env_->getSetpoint(), env_->getProcessValue(), control_signal});
                
                // Print every 100 steps
                if (step % 100 == 0 || step == num_steps - 1) {
                    std::cout << "Step " << step 
                              << ": PV=" << std::fixed << std::setprecision(6) << env_->getProcessValue() 
                              << ", Error=" << error
                              << ", Control=" << control_signal
                              << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during controller run: " << e.what() << std::endl;
        }
        
        return history;
    }

    /**
     * @brief Save the RL agent model
     * 
     * @param filename File to save to
     */
    void saveModel(const std::string& filename) {
        rl_agent_.saveModel(filename);
    }

    /**
     * @brief Load an RL agent model
     * 
     * @param filename File to load from
     */
    void loadModel(const std::string& filename) {
        try {
            rl_agent_.loadModel(filename);
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw; // Re-throw to notify caller
        }
    }

private:
    std::shared_ptr<Environment> env_;             // Environment to control
    std::shared_ptr<PIDController> pid_controller_; // PID controller
    RLAgent rl_agent_;                             // RL agent for parameter tuning
    std::atomic<bool> is_training_;                // Flag to indicate training status
};

/**
 * @brief Test the self-tuning PID controller
 */
void test_self_tuning_pid() {
    try {
        std::cout << "====== Starting Self-Tuning PID Controller Test ======" << std::endl;
        
        // Create environment with moderate complexity
        // Parameters: time_constant, gain, dt, disturbance_std_dev
        auto env = std::make_shared<SimpleProcessEnvironment>(2.0, 1.0, 0.01, 0.01);
        
        // Create PID controller with conservative initial parameters
        // This gives the training a reasonable starting point
        auto pid = std::make_shared<PIDController>(0.5, 0.1, 0.05, 0.01);
        
        // Create self-tuning PID system with larger hidden layer (32 neurons)
        SelfTuningPIDController self_tuning_pid(env, pid, 32);
        
        // See if a trained model exists and load it
        bool model_loaded = false;
        try {
            self_tuning_pid.loadModel("best_pid_model.dat");
            std::cout << "Loaded existing model. Current PID parameters: ";
            auto params = pid->getParameters();
            std::cout << "[" << params[0] << ", " << params[1] << ", " << params[2] << "]" << std::endl;
            model_loaded = true;
        } catch (const std::exception& e) {
            std::cout << "No existing model found, starting fresh training." << std::endl;
        }
        
        // Ask user which method to use
        std::cout << "Choose tuning method:" << std::endl;
        std::cout << "1. Reinforcement Learning (may be unstable)" << std::endl;
        std::cout << "2. Simple Grid Search (more reliable)" << std::endl;
        std::cout << "3. Skip training and use current parameters" << std::endl;
        std::cout << "Your choice (1-3): ";
        
        int method_choice = 2;  // Default to grid search
        std::cin >> method_choice;
        
        if (method_choice == 1) {
            // Training options
            int num_episodes = 1;
            std::cout << "Number of episodes to train (default 1): ";
            std::cin >> num_episodes;
            
            // Train the controller 
            std::cout << "Training the controller for " << num_episodes << " episodes..." << std::endl;
            self_tuning_pid.train(num_episodes, 1, 1);  // Sync and print every episode
        } else if (method_choice == 2) {
            // Use grid search instead of RL
            std::cout << "Performing grid search for PID parameters..." << std::endl;
            self_tuning_pid.simpleTuning(200);  // 200 steps per parameter set
        } else {
            std::cout << "Using current parameters without training." << std::endl;
        }
        
        // Test the controller with a setpoint change
        std::cout << "\nTesting the controller with different setpoints..." << std::endl;
        
        // Test with step response (sudden setpoint change)
        auto history1 = self_tuning_pid.run(300, 1.0);
        
        // Test with another setpoint to see adaptability
        auto history2 = self_tuning_pid.run(300, 0.5);
        
        // Calculate performance metrics only if we have data
        if (!history1.empty() && !history2.empty()) {
            // Print comparative results
            std::cout << "\nTest Results (Step Response to Setpoint = 1.0):" << std::endl;
            std::cout << "Time, Setpoint, Process Value, Control Signal" << std::endl;
            for (size_t i = 0; i < history1.size(); i += 50) {  // Print every 50th point
                std::cout << history1[i][0] << ", " 
                        << history1[i][1] << ", " 
                        << history1[i][2] << ", " 
                        << history1[i][3] << std::endl;
            }
            
            std::cout << "\nTest Results (Step Response to Setpoint = 0.5):" << std::endl;
            std::cout << "Time, Setpoint, Process Value, Control Signal" << std::endl;
            for (size_t i = 0; i < history2.size(); i += 50) {  // Print every 50th point
                std::cout << history2[i][0] << ", " 
                        << history2[i][1] << ", " 
                        << history2[i][2] << ", " 
                        << history2[i][3] << std::endl;
            }
            
            // Analyze performance
            double mse1 = 0.0, mse2 = 0.0;
            for (const auto& point : history1) {
                double error = point[1] - point[2]; // setpoint - process_value
                mse1 += error * error;
            }
            mse1 /= history1.size();
            
            for (const auto& point : history2) {
                double error = point[1] - point[2]; // setpoint - process_value
                mse2 += error * error;
            }
            mse2 /= history2.size();
            
            std::cout << "\nPerformance Analysis:" << std::endl;
            std::cout << "Mean Squared Error (Setpoint = 1.0): " << mse1 << std::endl;
            std::cout << "Mean Squared Error (Setpoint = 0.5): " << mse2 << std::endl;
        } else {
            std::cout << "No test data collected. Controller may have failed during testing." << std::endl;
        }
        
        // Final PID parameters
        auto final_params = pid->getParameters();
        std::cout << "\nFinal PID parameters: ["
                  << std::fixed << std::setprecision(6) << final_params[0] << ", "
                  << final_params[1] << ", "
                  << final_params[2] << "]" << std::endl;
        
        // Save the trained model
        if (method_choice != 3) {  // Only save if we did some training
            self_tuning_pid.saveModel("trained_pid_model.dat");
            std::cout << "Model saved to trained_pid_model.dat" << std::endl;
        }
        
        std::cout << "====== Self-Tuning PID Controller Test Completed ======" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
    }
}

int main() {
    // Set up signal handlers
    signal(SIGSEGV, signal_handler); // Segmentation fault
    signal(SIGABRT, signal_handler); // Abort
    signal(SIGFPE, signal_handler);  // Floating point exception
    signal(SIGTERM, signal_handler); // Termination
    signal(SIGINT, signal_handler);  // Interrupt (Ctrl+C)
    
    try {
        test_self_tuning_pid();
    } catch (const std::exception& e) {
        std::cerr << "Main error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred in main" << std::endl;
        return 2;
    }
    
    return 0;
}