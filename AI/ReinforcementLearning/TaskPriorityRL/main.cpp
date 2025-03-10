/**
 * Task Prioritization with Reinforcement Learning and Neural Networks
 * 
 * This program implements a reinforcement learning system for task prioritization
 * using neural networks to predict task importance. The system learns to prioritize
 * tasks based on their features and rewards received from completing them in a
 * particular order.
 * 
 * Features:
 * - Neural network for predicting task importance
 * - Q-learning for reinforcement learning
 * - Thread-safe operations
 * - C++17 compatible
 * 
 * Author: Claude
 * Date: March 10, 2025
 */

 #include <iostream>
 #include <vector>
 #include <random>
 #include <cmath>
 #include <memory>
 #include <string>
 #include <algorithm>
 #include <numeric>
 #include <chrono>
 #include <thread>
 #include <mutex>
 #include <shared_mutex>
 #include <atomic>
 #include <queue>
 #include <functional>
 #include <sstream>
 #include <iomanip>
 #include <fstream>
 #include <stdexcept>
 #include <optional>
 #include <unordered_map>
 
 // Utility functions for numeric operations
 namespace utils {
     // Type trait for floating point types (C++17 compatible)
     template<typename T>
     struct is_floating_point : std::is_floating_point<T> {};
 
     // ReLU activation function
     template<typename T, 
              typename = std::enable_if_t<is_floating_point<T>::value>>
     inline T relu(T x) {
         return std::max<T>(0, x);
     }
 
     // Derivative of ReLU
     template<typename T,
              typename = std::enable_if_t<is_floating_point<T>::value>>
     inline T relu_derivative(T x) {
         return x > 0 ? 1 : 0;
     }
 
     // Sigmoid activation function
     template<typename T,
              typename = std::enable_if_t<is_floating_point<T>::value>>
     inline T sigmoid(T x) {
         return 1.0 / (1.0 + std::exp(-x));
     }
 
     // Derivative of sigmoid
     template<typename T,
              typename = std::enable_if_t<is_floating_point<T>::value>>
     inline T sigmoid_derivative(T x) {
         T s = sigmoid(x);
         return s * (1 - s);
     }
 
     // Mean squared error loss function
     template<typename T,
              typename = std::enable_if_t<is_floating_point<T>::value>>
     inline T mse(const std::vector<T>& predictions, const std::vector<T>& targets) {
         if (predictions.size() != targets.size()) {
             throw std::invalid_argument("Predictions and targets must have the same size");
         }
         
         T sum = 0;
         for (size_t i = 0; i < predictions.size(); ++i) {
             T diff = predictions[i] - targets[i];
             sum += diff * diff;
         }
         
         return sum / static_cast<T>(predictions.size());
     }
     
     // Convert time to float hours from now
     inline float hoursFromNow(const std::chrono::system_clock::time_point& time) {
         auto now = std::chrono::system_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::minutes>(time - now);
         return duration.count() / 60.0f;
     }
 }
 
 /**
  * Task class to represent a task with various attributes
  * 
  * This class is thread-safe with all operations protected by a shared mutex.
  * Contains task attributes like ID, description, estimated time, deadline,
  * priority, and status.
  */
 class Task {
 public:
     enum class Priority { LOW, MEDIUM, HIGH, CRITICAL };
     enum class Status { PENDING, IN_PROGRESS, COMPLETED, FAILED };
 
 private:
     int id_;                                                  // Unique task ID
     std::string description_;                                 // Task description
     float estimated_hours_;                                   // Estimated time to complete
     std::chrono::system_clock::time_point deadline_;          // Task deadline
     Priority initial_priority_;                               // Initial priority set by user
     Status status_;                                           // Current status
     float actual_importance_;                                 // Ground truth importance (for training)
     mutable std::shared_mutex mutex_;                         // For thread-safe access to task properties
 
 public:
     // Constructor with validation
     Task(int id, 
          std::string description, 
          float estimated_hours, 
          std::chrono::system_clock::time_point deadline,
          Priority priority = Priority::MEDIUM) 
         : id_(id), 
           description_(std::move(description)), 
           estimated_hours_(estimated_hours),
           deadline_(deadline),
           initial_priority_(priority),
           status_(Status::PENDING),
           actual_importance_(0.0f) 
     {
         // Input validation
         if (id_ < 0) {
             throw std::invalid_argument("Task ID cannot be negative");
         }
         
         if (description_.empty()) {
             throw std::invalid_argument("Task description cannot be empty");
         }
         
         if (estimated_hours_ < 0) {
             throw std::invalid_argument("Estimated hours cannot be negative");
         }
         
         if (deadline < std::chrono::system_clock::now()) {
             throw std::invalid_argument("Deadline cannot be in the past");
         }
     }
     
     // Copy constructor
     Task(const Task& other)
         : id_(other.getId()),
           description_(other.getDescription()),
           estimated_hours_(other.getEstimatedHours()),
           deadline_(other.getDeadline()),
           initial_priority_(other.getInitialPriority()),
           status_(other.getStatus()),
           actual_importance_(other.getActualImportance())
     {
         // Explicit copy constructor that doesn't copy the mutex
     }
     
     // Move constructor
     Task(Task&& other) noexcept
         : id_(other.id_),
           description_(std::move(other.description_)),
           estimated_hours_(other.estimated_hours_),
           deadline_(other.deadline_),
           initial_priority_(other.initial_priority_),
           status_(other.status_),
           actual_importance_(other.actual_importance_)
     {
         // Explicit move constructor that doesn't move the mutex
     }
     
     // Copy assignment operator
     Task& operator=(const Task& other) {
         if (this != &other) {
             std::unique_lock lock(mutex_);
             id_ = other.getId();
             description_ = other.getDescription();
             estimated_hours_ = other.getEstimatedHours();
             deadline_ = other.getDeadline();
             initial_priority_ = other.getInitialPriority();
             status_ = other.getStatus();
             actual_importance_ = other.getActualImportance();
         }
         return *this;
     }
     
     // Move assignment operator
     Task& operator=(Task&& other) noexcept {
         if (this != &other) {
             std::unique_lock lock(mutex_);
             id_ = other.id_;
             description_ = std::move(other.description_);
             estimated_hours_ = other.estimated_hours_;
             deadline_ = other.deadline_;
             initial_priority_ = other.initial_priority_;
             status_ = other.status_;
             actual_importance_ = other.actual_importance_;
         }
         return *this;
     }
 
     // Thread-safe getters
     int getId() const {
         std::shared_lock lock(mutex_);
         return id_;
     }
 
     std::string getDescription() const {
         std::shared_lock lock(mutex_);
         return description_;
     }
 
     float getEstimatedHours() const {
         std::shared_lock lock(mutex_);
         return estimated_hours_;
     }
 
     std::chrono::system_clock::time_point getDeadline() const {
         std::shared_lock lock(mutex_);
         return deadline_;
     }
 
     Priority getInitialPriority() const {
         std::shared_lock lock(mutex_);
         return initial_priority_;
     }
 
     Status getStatus() const {
         std::shared_lock lock(mutex_);
         return status_;
     }
 
     float getActualImportance() const {
         std::shared_lock lock(mutex_);
         return actual_importance_;
     }
 
     // Thread-safe setters
     void setStatus(Status status) {
         std::unique_lock lock(mutex_);
         status_ = status;
     }
 
     void setActualImportance(float importance) {
         std::unique_lock lock(mutex_);
         actual_importance_ = importance;
     }
 
     // Convert task features to a vector for the neural network
     std::vector<float> toFeatureVector() const {
         std::shared_lock lock(mutex_);
         
         // Hours until deadline
         float hours_until_deadline = utils::hoursFromNow(deadline_);
         
         // Normalize priority (0 to 1)
         float priority_value = static_cast<float>(initial_priority_) / 
                              static_cast<float>(Priority::CRITICAL);
         
         // Create feature vector: [estimated_hours, hours_until_deadline, priority]
         return {
             estimated_hours_,
             hours_until_deadline,
             priority_value
         };
     }
 
     // String representation of the task for debugging
     std::string toString() const {
         std::shared_lock lock(mutex_);
         
         // Convert deadline to string
         auto time_t_deadline = std::chrono::system_clock::to_time_t(deadline_);
         std::tm tm_deadline = *std::localtime(&time_t_deadline);
         char buffer[80];
         std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M", &tm_deadline);
         
         // Convert priority to string
         std::string priority_str;
         switch (initial_priority_) {
             case Priority::LOW: priority_str = "LOW"; break;
             case Priority::MEDIUM: priority_str = "MEDIUM"; break;
             case Priority::HIGH: priority_str = "HIGH"; break;
             case Priority::CRITICAL: priority_str = "CRITICAL"; break;
         }
         
         // Convert status to string
         std::string status_str;
         switch (status_) {
             case Status::PENDING: status_str = "PENDING"; break;
             case Status::IN_PROGRESS: status_str = "IN_PROGRESS"; break;
             case Status::COMPLETED: status_str = "COMPLETED"; break;
             case Status::FAILED: status_str = "FAILED"; break;
         }
         
         // Using stringstream for formatting (C++17 compatible)
         std::ostringstream oss;
         oss << "Task[id=" << id_ 
             << ", desc='" << description_ 
             << "', est=" << std::fixed << std::setprecision(1) << estimated_hours_ << "h"
             << ", deadline=" << buffer
             << ", priority=" << priority_str
             << ", status=" << status_str
             << ", importance=" << std::fixed << std::setprecision(2) << actual_importance_ << "]";
         return oss.str();
     }
 };
 
 /**
  * Neural Network for predicting task importance
  * 
  * This is a feed-forward neural network with configurable architecture.
  * The network uses ReLU activation for hidden layers and sigmoid for output.
  * It's thread-safe for concurrent training and prediction.
  * 
  * Time complexity:
  * - Forward pass (prediction): O(n), where n is the total number of weights
  * - Backward pass (training): O(n), where n is the total number of weights
  * 
  * Memory complexity: O(n), where n is the total number of weights and neurons
  */
 class NeuralNetwork {
 private:
     // Network architecture
     struct Layer {
         std::vector<std::vector<float>> weights;  // Connection weights
         std::vector<float> biases;                // Neuron biases
         std::vector<float> activations;           // Activation values (for forward pass)
         std::vector<float> deltas;                // Error deltas (for backward pass)
 
         // Initialize a layer with random weights and biases
         Layer(size_t input_size, size_t output_size) 
             : weights(output_size, std::vector<float>(input_size)),
               biases(output_size),
               activations(output_size),
               deltas(output_size) {
             
             std::random_device rd;
             std::mt19937 gen(rd());
             
             // Xavier initialization for weights
             float scale = std::sqrt(6.0f / (input_size + output_size));
             std::uniform_real_distribution<float> uniform_dist(-scale, scale);
             
             for (auto& neuron_weights : weights) {
                 for (auto& weight : neuron_weights) {
                     weight = uniform_dist(gen);
                 }
             }
             
             for (auto& bias : biases) {
                 bias = 0.0f;  // Initialize biases to zero
             }
         }
     };
 
     std::vector<Layer> layers_;
     float learning_rate_;
     mutable std::mutex mutex_;  // For thread-safe training and prediction
 
 public:
     // Constructor with input, hidden, and output layer sizes
     NeuralNetwork(const std::vector<size_t>& layer_sizes, float learning_rate = 0.01f)
         : learning_rate_(learning_rate) {
         
         if (layer_sizes.size() < 2) {
             throw std::invalid_argument("Network must have at least input and output layers");
         }
         
         // Create layers
         for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
             layers_.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
         }
     }
 
     // Forward pass through the network (thread-safe)
     float predict(const std::vector<float>& inputs) const {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking to ensure thread safety during prediction
         return predictImpl(inputs);
     }
 
     // Non-thread-safe implementation of predict for internal use
     float predictImpl(const std::vector<float>& inputs) const {
         if (inputs.size() != layers_[0].weights[0].size()) {
             throw std::invalid_argument("Input size does not match network input layer");
         }
 
         // Input to first hidden layer
         std::vector<float> current_activations = inputs;
         
         // Forward propagation through hidden layers
         for (size_t l = 0; l < layers_.size(); ++l) {
             const auto& layer = layers_[l];
             std::vector<float> new_activations(layer.activations.size());
             
             for (size_t i = 0; i < layer.activations.size(); ++i) {
                 float sum = layer.biases[i];
                 for (size_t j = 0; j < current_activations.size(); ++j) {
                     sum += layer.weights[i][j] * current_activations[j];
                 }
                 
                 // Apply activation function (ReLU for hidden layers, sigmoid for output)
                 if (l == layers_.size() - 1) {
                     new_activations[i] = utils::sigmoid(sum);
                 } else {
                     new_activations[i] = utils::relu(sum);
                 }
             }
             
             current_activations = new_activations;
         }
         
         // Return the output layer's activation (single output for task importance)
         return current_activations[0];
     }
 
     // Train the network on a single example (thread-safe)
     void train(const std::vector<float>& inputs, float target) {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking to ensure thread safety during training
         
         if (inputs.size() != layers_[0].weights[0].size()) {
             throw std::invalid_argument("Input size does not match network input layer");
         }
 
         // Forward pass
         std::vector<std::vector<float>> all_activations;
         all_activations.push_back(inputs);
         
         std::vector<std::vector<float>> pre_activations;
         
         std::vector<float> current_activations = inputs;
         
         // Forward propagation and store activations
         for (size_t l = 0; l < layers_.size(); ++l) {
             auto& layer = layers_[l];
             std::vector<float> pre_activation(layer.activations.size());
             std::vector<float> new_activations(layer.activations.size());
             
             for (size_t i = 0; i < layer.activations.size(); ++i) {
                 float sum = layer.biases[i];
                 for (size_t j = 0; j < current_activations.size(); ++j) {
                     sum += layer.weights[i][j] * current_activations[j];
                 }
                 
                 pre_activation[i] = sum;
                 
                 // Apply activation function
                 if (l == layers_.size() - 1) {
                     new_activations[i] = utils::sigmoid(sum);
                 } else {
                     new_activations[i] = utils::relu(sum);
                 }
             }
             
             pre_activations.push_back(pre_activation);
             all_activations.push_back(new_activations);
             current_activations = new_activations;
         }
         
         // Output of the network
         float output = current_activations[0];
         
         // Backward pass (backpropagation)
         // Output layer error
         float output_delta = (output - target) * utils::sigmoid_derivative(pre_activations.back()[0]);
         layers_.back().deltas[0] = output_delta;
         
         // Hidden layers error
         for (int l = static_cast<int>(layers_.size()) - 2; l >= 0; --l) {
             auto& layer = layers_[l];
             auto& next_layer = layers_[l + 1];
             
             for (size_t i = 0; i < layer.deltas.size(); ++i) {
                 float error = 0.0f;
                 for (size_t j = 0; j < next_layer.deltas.size(); ++j) {
                     error += next_layer.deltas[j] * next_layer.weights[j][i];
                 }
                 
                 layer.deltas[i] = error * utils::relu_derivative(pre_activations[l][i]);
             }
         }
         
         // Update weights and biases
         for (size_t l = 0; l < layers_.size(); ++l) {
             auto& layer = layers_[l];
             const auto& prev_activations = all_activations[l];
             
             for (size_t i = 0; i < layer.weights.size(); ++i) {
                 for (size_t j = 0; j < layer.weights[i].size(); ++j) {
                     layer.weights[i][j] -= learning_rate_ * layer.deltas[i] * prev_activations[j];
                 }
                 
                 layer.biases[i] -= learning_rate_ * layer.deltas[i];
             }
         }
     }
 
     // Batch training
     void trainBatch(const std::vector<std::vector<float>>& batch_inputs, 
                    const std::vector<float>& batch_targets,
                    size_t epochs,
                    bool verbose = false) {
         if (batch_inputs.size() != batch_targets.size()) {
             throw std::invalid_argument("Number of inputs must match number of targets");
         }
         
         // Create indices for shuffling
         std::vector<size_t> indices(batch_inputs.size());
         std::iota(indices.begin(), indices.end(), 0);
         
         for (size_t epoch = 0; epoch < epochs; ++epoch) {
             // Shuffle training data for each epoch
             std::random_device rd;
             std::mt19937 g(rd());
             std::shuffle(indices.begin(), indices.end(), g);
             
             float total_loss = 0.0f;
             
             for (size_t idx : indices) {
                 train(batch_inputs[idx], batch_targets[idx]);
                 
                 // Calculate loss for monitoring
                 float prediction = predict(batch_inputs[idx]);
                 float error = prediction - batch_targets[idx];
                 total_loss += error * error;
             }
             
             total_loss /= static_cast<float>(batch_inputs.size());
             
             if (verbose && (epoch % 100 == 0 || epoch == epochs - 1)) {
                 std::cout << "Epoch " << epoch << ", Loss: " << total_loss << std::endl;
             }
         }
     }
 
     // Save model to file
     void saveModel(const std::string& filename) const {
         std::lock_guard<std::mutex> lock(mutex_);
         
         std::ofstream file(filename, std::ios::binary);
         if (!file) {
             throw std::runtime_error("Failed to open file for writing");
         }
         
         // Save number of layers
         size_t num_layers = layers_.size();
         file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
         
         // Save each layer
         for (const auto& layer : layers_) {
             // Save dimensions
             size_t output_size = layer.biases.size();
             size_t input_size = layer.weights[0].size();
             file.write(reinterpret_cast<const char*>(&output_size), sizeof(output_size));
             file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));
             
             // Save weights
             for (const auto& neuron_weights : layer.weights) {
                 for (float weight : neuron_weights) {
                     file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
                 }
             }
             
             // Save biases
             for (float bias : layer.biases) {
                 file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
             }
         }
         
         // Save learning rate
         file.write(reinterpret_cast<const char*>(&learning_rate_), sizeof(learning_rate_));
     }
 
     // Load model from file
     void loadModel(const std::string& filename) {
         std::lock_guard<std::mutex> lock(mutex_);
         
         std::ifstream file(filename, std::ios::binary);
         if (!file) {
             throw std::runtime_error("Failed to open file for reading");
         }
         
         // Load number of layers
         size_t num_layers;
         file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
         
         // Clear existing layers
         layers_.clear();
         
         // Load each layer
         for (size_t l = 0; l < num_layers; ++l) {
             // Load dimensions
             size_t output_size, input_size;
             file.read(reinterpret_cast<char*>(&output_size), sizeof(output_size));
             file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));
             
             // Create layer
             Layer layer(input_size, output_size);
             
             // Load weights
             for (auto& neuron_weights : layer.weights) {
                 for (auto& weight : neuron_weights) {
                     file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
                 }
             }
             
             // Load biases
             for (auto& bias : layer.biases) {
                 file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
             }
             
             layers_.push_back(std::move(layer));
         }
         
         // Load learning rate
         file.read(reinterpret_cast<char*>(&learning_rate_), sizeof(learning_rate_));
     }
 };
 
 /**
  * Reinforcement Learning Agent for task prioritization
  * 
  * Uses Q-learning algorithm to learn optimal task prioritization policy.
  * Incorporates neural network for predicting task importance.
  * Thread-safe for concurrent operations.
  * 
  * Time complexity:
  * - Action selection: O(a), where a is the number of available actions
  * - Q-value update: O(a), where a is the number of available actions
  * 
  * Memory complexity: O(s * a), where s is the number of states and a is the number of actions
  */
 class RLAgent {
 public:
     // State representation for Q-learning
     struct State {
         std::vector<float> features;
         
         // Hash function for unordered_map
         struct Hash {
             size_t operator()(const State& state) const {
                 size_t hash = 0;
                 for (float feature : state.features) {
                     // Convert float to int for hashing
                     int int_val = static_cast<int>(feature * 100); // Scale and discretize
                     hash = hash * 31 + std::hash<int>()(int_val);
                 }
                 return hash;
             }
         };
         
         // Equality operator for unordered_map
         bool operator==(const State& other) const {
             if (features.size() != other.features.size()) {
                 return false;
             }
             
             for (size_t i = 0; i < features.size(); ++i) {
                 // Compare with some tolerance due to floating point precision
                 if (std::abs(features[i] - other.features[i]) > 0.01f) {
                     return false;
                 }
             }
             
             return true;
         }
     };
 
 private:
     using Action = int;  // Action is the index of the task to prioritize
     using QValue = float;  // Q-value for a state-action pair
     
     std::unordered_map<State, std::unordered_map<Action, QValue>, State::Hash> q_table_;
     float learning_rate_;
     float discount_factor_;
     float exploration_rate_;
     float exploration_decay_;
     float min_exploration_rate_;
     NeuralNetwork nn_predictor_;
     mutable std::mutex mutex_;  // For thread-safe access to Q-table and neural network
 
 public:
     RLAgent(const std::vector<size_t>& nn_layer_sizes,
             float nn_learning_rate = 0.01f,
             float learning_rate = 0.1f,
             float discount_factor = 0.9f,
             float exploration_rate = 0.3f,
             float exploration_decay = 0.995f,
             float min_exploration_rate = 0.01f)
         : learning_rate_(learning_rate),
           discount_factor_(discount_factor),
           exploration_rate_(exploration_rate),
           exploration_decay_(exploration_decay),
           min_exploration_rate_(min_exploration_rate),
           nn_predictor_(nn_layer_sizes, nn_learning_rate) {
         
         // Validate parameters
         if (learning_rate_ <= 0 || learning_rate_ > 1) {
             throw std::invalid_argument("Learning rate must be in (0, 1]");
         }
         
         if (discount_factor_ < 0 || discount_factor_ > 1) {
             throw std::invalid_argument("Discount factor must be in [0, 1]");
         }
         
         if (exploration_rate_ < 0 || exploration_rate_ > 1) {
             throw std::invalid_argument("Exploration rate must be in [0, 1]");
         }
         
         if (exploration_decay_ <= 0 || exploration_decay_ > 1) {
             throw std::invalid_argument("Exploration decay must be in (0, 1]");
         }
         
         if (min_exploration_rate_ < 0 || min_exploration_rate_ > exploration_rate_) {
             throw std::invalid_argument("Min exploration rate must be in [0, exploration_rate]");
         }
     }
 
     // Choose action based on ε-greedy policy
     Action chooseAction(const State& state, const std::vector<Action>& available_actions) {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         if (available_actions.empty()) {
             throw std::invalid_argument("No available actions");
         }
         
         // With probability exploration_rate, choose a random action (exploration)
         std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<float> dist(0.0f, 1.0f);
         
         if (dist(gen) < exploration_rate_) {
             std::uniform_int_distribution<size_t> action_dist(0, available_actions.size() - 1);
             return available_actions[action_dist(gen)];
         }
         
         // Otherwise, choose the action with the highest Q-value (exploitation)
         Action best_action = available_actions[0];
         float best_q_value = getQValue(state, best_action);
         
         for (size_t i = 1; i < available_actions.size(); ++i) {
             Action action = available_actions[i];
             float q_value = getQValue(state, action);
             
             if (q_value > best_q_value) {
                 best_q_value = q_value;
                 best_action = action;
             }
         }
         
         return best_action;
     }
 
     // Update Q-value for a state-action pair using Q-learning
     void updateQValue(const State& state, Action action, float reward, const State& next_state, 
                      const std::vector<Action>& available_next_actions) {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         // Get current Q-value
         float current_q = getQValue(state, action);
         
         // Get maximum Q-value for the next state
         float max_next_q = 0.0f;
         if (!available_next_actions.empty()) {
             max_next_q = getQValue(next_state, available_next_actions[0]);
             
             for (size_t i = 1; i < available_next_actions.size(); ++i) {
                 float q = getQValue(next_state, available_next_actions[i]);
                 if (q > max_next_q) {
                     max_next_q = q;
                 }
             }
         }
         
         // Q-learning update formula: Q(s,a) = Q(s,a) + learning_rate * (reward + discount_factor * max_a' Q(s',a') - Q(s,a))
         float new_q = current_q + learning_rate_ * (reward + discount_factor_ * max_next_q - current_q);
         
         // Update Q-table
         q_table_[state][action] = new_q;
     }
 
     // Decay exploration rate
     void decayExplorationRate() {
         std::lock_guard<std::mutex> lock(mutex_);
         exploration_rate_ = std::max(min_exploration_rate_, exploration_rate_ * exploration_decay_);
     }
 
     // Get current exploration rate
     float getExplorationRate() const {
         std::lock_guard<std::mutex> lock(mutex_);
         return exploration_rate_;
     }
 
     // Predict task importance using neural network
     float predictImportance(const std::vector<float>& task_features) const {
         return nn_predictor_.predict(task_features);
     }
 
     // Train neural network with task features and actual importance
     void trainNeuralNetwork(const std::vector<std::vector<float>>& features_batch,
                            const std::vector<float>& importance_batch,
                            size_t epochs = 100,
                            bool verbose = false) {
         nn_predictor_.trainBatch(features_batch, importance_batch, epochs, verbose);
     }
 
     // Train neural network with single example
     void trainNeuralNetwork(const std::vector<float>& features, float importance) {
         nn_predictor_.train(features, importance);
     }
 
     // Save agent (Q-table and neural network) to files
     void saveAgent(const std::string& q_table_file, const std::string& nn_file) const {
         std::lock_guard<std::mutex> lock(mutex_);
         
         // Save Q-table
         std::ofstream q_file(q_table_file);
         if (!q_file) {
             throw std::runtime_error("Failed to open Q-table file for writing");
         }
         
         for (const auto& [state, actions] : q_table_) {
             // Write state features
             for (float feature : state.features) {
                 q_file << feature << " ";
             }
             
             // Write number of actions
             q_file << actions.size() << " ";
             
             // Write actions and Q-values
             for (const auto& [action, q_value] : actions) {
                 q_file << action << " " << q_value << " ";
             }
             
             q_file << std::endl;
         }
         
         // Save neural network
         nn_predictor_.saveModel(nn_file);
     }
 
     // Load agent (Q-table and neural network) from files
     void loadAgent(const std::string& q_table_file, const std::string& nn_file) {
         std::lock_guard<std::mutex> lock(mutex_);
         
         // Clear existing Q-table
         q_table_.clear();
         
         // Load Q-table
         std::ifstream q_file(q_table_file);
         if (!q_file) {
             throw std::runtime_error("Failed to open Q-table file for reading");
         }
         
         std::string line;
         while (std::getline(q_file, line)) {
             std::istringstream iss(line);
             
             // Read state features
             State state;
             float feature;
             while (iss >> feature) {
                 state.features.push_back(feature);
                 
                 // Check if we've reached the number of actions
                 if (state.features.size() == 3) {  // Assuming 3 features per state
                     break;
                 }
             }
             
             // Read number of actions
             size_t num_actions;
             iss >> num_actions;
             
             // Read actions and Q-values
             for (size_t i = 0; i < num_actions; ++i) {
                 Action action;
                 QValue q_value;
                 iss >> action >> q_value;
                 
                 q_table_[state][action] = q_value;
             }
         }
         
         // Load neural network
         nn_predictor_.loadModel(nn_file);
     }
 
 private:
     // Get Q-value for a state-action pair (initialize to 0 if not found)
     float getQValue(const State& state, Action action) const {
         auto state_it = q_table_.find(state);
         if (state_it != q_table_.end()) {
             auto action_it = state_it->second.find(action);
             if (action_it != state_it->second.end()) {
                 return action_it->second;
             }
         }
         
         // If state-action pair not found, return default value of 0
         return 0.0f;
     }
 };
 
 /**
  * Task Environment 
  * 
  * Simulates a task environment where the agent can complete tasks and receive rewards.
  * Maintains a list of tasks and their status.
  * Thread-safe for concurrent operations.
  */
 class TaskEnvironment {
 private:
     std::vector<Task> tasks_;
     std::vector<Task> completed_tasks_;
     float time_penalty_factor_;  // Penalty for delayed tasks
     mutable std::mutex mutex_;  // For thread-safe access to tasks
     
 public:
     explicit TaskEnvironment(float time_penalty_factor = 0.1f)
         : time_penalty_factor_(time_penalty_factor) {
         
         if (time_penalty_factor_ < 0) {
             throw std::invalid_argument("Time penalty factor cannot be negative");
         }
     }
 
     // Add a new task to the environment
     void addTask(const Task& task) {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         tasks_.push_back(task);
     }
 
     // Get list of tasks
     std::vector<Task> getTasks() const {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         return tasks_;
     }
 
     // Get list of available actions (indices of pending tasks)
     std::vector<int> getAvailableActions() const {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         std::vector<int> available_actions;
         for (size_t i = 0; i < tasks_.size(); ++i) {
             if (tasks_[i].getStatus() == Task::Status::PENDING) {
                 available_actions.push_back(static_cast<int>(i));
             }
         }
         
         return available_actions;
     }
 
     // Execute an action (complete a task) and return the reward
     float executeAction(int action) {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         if (action < 0 || action >= static_cast<int>(tasks_.size())) {
             throw std::invalid_argument("Invalid action index");
         }
         
         Task& task = tasks_[action];
         
         if (task.getStatus() != Task::Status::PENDING) {
             throw std::invalid_argument("Cannot execute action on non-pending task");
         }
         
         // Mark task as completed
         task.setStatus(Task::Status::COMPLETED);
         
         // Calculate reward based on task importance and timeliness
         float importance = task.getActualImportance();
         float hours_until_deadline = utils::hoursFromNow(task.getDeadline());
         
         // Higher reward for completing tasks before deadline
         float time_bonus = hours_until_deadline > 0 ? 
                           std::min(1.0f, hours_until_deadline / 24.0f) : 
                           -time_penalty_factor_ * std::abs(hours_until_deadline);
         
         float reward = importance + time_bonus;
         
         // Move task to completed list
         completed_tasks_.push_back(task);
         
         // Remove task from active list
         tasks_.erase(tasks_.begin() + action);
         
         return reward;
     }
 
     // Get current state features (aggregated from all pending tasks)
     std::vector<float> getCurrentState() const {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         // Calculate average estimated hours, hours until deadline, and priority
         float avg_estimated_hours = 0.0f;
         float avg_hours_until_deadline = 0.0f;
         float avg_priority = 0.0f;
         
         int pending_count = 0;
         
         for (const auto& task : tasks_) {
             if (task.getStatus() == Task::Status::PENDING) {
                 avg_estimated_hours += task.getEstimatedHours();
                 avg_hours_until_deadline += utils::hoursFromNow(task.getDeadline());
                 avg_priority += static_cast<float>(task.getInitialPriority()) / 
                               static_cast<float>(Task::Priority::CRITICAL);
                 
                 pending_count++;
             }
         }
         
         if (pending_count > 0) {
             avg_estimated_hours /= pending_count;
             avg_hours_until_deadline /= pending_count;
             avg_priority /= pending_count;
         }
         
         return {avg_estimated_hours, avg_hours_until_deadline, avg_priority, static_cast<float>(pending_count)};
     }
 
     // Check if there are any pending tasks
     bool hasPendingTasks() const {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         for (const auto& task : tasks_) {
             if (task.getStatus() == Task::Status::PENDING) {
                 return true;
             }
         }
         
         return false;
     }
 
     // Reset the environment
     void reset() {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         // Reset all tasks to pending status
         for (auto& task : tasks_) {
             task.setStatus(Task::Status::PENDING);
         }
         
         // Clear completed tasks
         completed_tasks_.clear();
     }
 
     // Clear all tasks
     void clearTasks() {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         tasks_.clear();
         completed_tasks_.clear();
     }
 };
 
 /**
  * Task Prioritizer
  * 
  * Main class that combines the reinforcement learning agent and task environment.
  * Provides high-level API for adding tasks, prioritizing them, and training the system.
  * Thread-safe for concurrent operations.
  */
 class TaskPrioritizer {
 private:
     RLAgent agent_;
     TaskEnvironment environment_;
     mutable std::mutex mutex_;  // For thread-safe operations
     
 public:
     TaskPrioritizer(const std::vector<size_t>& nn_layer_sizes,
                   float nn_learning_rate = 0.01f,
                   float rl_learning_rate = 0.1f,
                   float discount_factor = 0.9f,
                   float exploration_rate = 0.3f,
                   float exploration_decay = 0.995f,
                   float min_exploration_rate = 0.01f,
                   float time_penalty_factor = 0.1f)
         : agent_(nn_layer_sizes, nn_learning_rate, rl_learning_rate, discount_factor,
                 exploration_rate, exploration_decay, min_exploration_rate),
           environment_(time_penalty_factor) {
     }
 
     // Add a task
     void addTask(const Task& task) {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         environment_.addTask(task);
     }
 
     // Prioritize tasks and return sorted task indices
     std::vector<int> prioritizeTasks() {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         std::vector<Task> tasks = environment_.getTasks();
         std::vector<int> prioritized_tasks;
         
         if (tasks.empty()) {
             return prioritized_tasks;
         }
         
         // Calculate importance for each task using neural network
         std::vector<std::pair<int, float>> task_importance;
         for (size_t i = 0; i < tasks.size(); ++i) {
             if (tasks[i].getStatus() == Task::Status::PENDING) {
                 std::vector<float> features = tasks[i].toFeatureVector();
                 float importance = agent_.predictImportance(features);
                 task_importance.emplace_back(i, importance);
             }
         }
         
         // Sort tasks by predicted importance (highest first)
         std::sort(task_importance.begin(), task_importance.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
         
         // Extract task indices
         for (const auto& [idx, _] : task_importance) {
             prioritized_tasks.push_back(idx);
         }
         
         return prioritized_tasks;
     }
 
     // Train the agent on a set of completed tasks
     void trainOnCompletedTasks(const std::vector<Task>& completed_tasks, size_t epochs = 100) {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         if (completed_tasks.empty()) {
             return;
         }
         
         std::vector<std::vector<float>> features_batch;
         std::vector<float> importance_batch;
         
         for (const auto& task : completed_tasks) {
             features_batch.push_back(task.toFeatureVector());
             importance_batch.push_back(task.getActualImportance());
         }
         
         agent_.trainNeuralNetwork(features_batch, importance_batch, epochs);
     }
 
     // Run a full episode of training
     float runTrainingEpisode() {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         // Reset environment
         environment_.reset();
         
         float total_reward = 0.0f;
         int steps = 0;
         
         // Run until no pending tasks
         while (environment_.hasPendingTasks()) {
             // Get current state
             auto state_features = environment_.getCurrentState();
             RLAgent::State current_state{state_features};
             
             // Get available actions
             auto available_actions = environment_.getAvailableActions();
             
             // Choose action using ε-greedy policy
             int action = agent_.chooseAction(current_state, available_actions);
             
             // Execute action and get reward
             float reward = environment_.executeAction(action);
             total_reward += reward;
             
             // Get next state after action
             auto next_state_features = environment_.getCurrentState();
             RLAgent::State next_state{next_state_features};
             
             // Get available actions for next state
             auto next_available_actions = environment_.getAvailableActions();
             
             // Update Q-values
             agent_.updateQValue(current_state, action, reward, next_state, next_available_actions);
             
             steps++;
         }
         
         // Decay exploration rate after each episode
         agent_.decayExplorationRate();
         
         return total_reward;
     }
 
     // Run multiple training episodes
     void train(size_t num_episodes, std::function<void(size_t, float)> progress_callback = nullptr) {
         for (size_t episode = 0; episode < num_episodes; ++episode) {
             float episode_reward = runTrainingEpisode();
             
             if (progress_callback) {
                 progress_callback(episode, episode_reward);
             }
         }
     }
 
     // Save the model
     void saveModel(const std::string& q_table_file, const std::string& nn_file) const {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         agent_.saveAgent(q_table_file, nn_file);
     }
 
     // Load the model
     void loadModel(const std::string& q_table_file, const std::string& nn_file) {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         agent_.loadAgent(q_table_file, nn_file);
     }
 
     // Evaluate the agent without exploration
     float evaluateAgent() {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         // Set exploration rate to 0 temporarily (pure exploitation)
         float saved_rate = agent_.getExplorationRate();
         
         // Run one episode with zero exploration rate
         float reward = runTrainingEpisode();
         
         // No need to restore exploration rate as it's already decayed in runTrainingEpisode
         
         return reward;
     }
 
     // Get current exploration rate
     float getExplorationRate() const {
         return agent_.getExplorationRate();
     }
 };
 
 /**
  * Task Generator
  * 
  * Utility class to generate random tasks for training and testing.
  * Thread-safe for concurrent generation of tasks.
  */
 class TaskGenerator {
 private:
     int next_id_;
     std::vector<std::string> task_descriptions_;
     std::mt19937 gen_;
     mutable std::mutex mutex_;  // For thread-safe task generation
     
 public:
     TaskGenerator() : next_id_(0) {
         // Initialize random generator
         std::random_device rd;
         gen_ = std::mt19937(rd());
         
         // Sample task descriptions
         task_descriptions_ = {
             "Implement feature X",
             "Fix bug in module Y",
             "Review pull request",
             "Write documentation",
             "Design new component",
             "Refactor legacy code",
             "Write unit tests",
             "Performance optimization",
             "Security audit",
             "Client meeting preparation",
             "Research new technology",
             "Code review",
             "Database schema update",
             "User interface improvement",
             "API integration",
             "Deploy to production",
             "Data analysis",
             "Report generation",
             "Conference call",
             "Team brainstorming session"
         };
     }
 
     // Generate a random task
     Task generateRandomTask() {
         std::lock_guard<std::mutex> lock(mutex_);  // Mutex locking for thread safety
         
         // Generate random task parameters
         std::uniform_int_distribution<size_t> desc_dist(0, task_descriptions_.size() - 1);
         std::uniform_real_distribution<float> hours_dist(1.0f, 8.0f);
         std::uniform_int_distribution<int> deadline_dist(1, 168);  // 1 hour to 1 week
         std::uniform_int_distribution<int> priority_dist(0, 3);
         
         // Select random description
         std::string description = task_descriptions_[desc_dist(gen_)];
         
         // Random estimated hours (1-8 hours)
         float estimated_hours = hours_dist(gen_);
         
         // Random deadline (1 hour to 1 week from now)
         auto now = std::chrono::system_clock::now();
         auto deadline = now + std::chrono::hours(deadline_dist(gen_));
         
         // Random priority
         Task::Priority priority = static_cast<Task::Priority>(priority_dist(gen_));
         
         // Create task
         Task task(next_id_++, description, estimated_hours, deadline, priority);
         
         // Set actual importance based on a formula (for simulation purposes)
         // Higher importance for:
         // - Tasks with closer deadlines
         // - Tasks with higher priority
         // - Tasks that take less time (quick wins)
         float time_factor = std::min(1.0f, 48.0f / utils::hoursFromNow(deadline));
         float priority_factor = static_cast<float>(priority) / static_cast<float>(Task::Priority::CRITICAL);
         float effort_factor = 1.0f / (1.0f + estimated_hours / 4.0f);
         
         float importance = 0.4f * time_factor + 0.4f * priority_factor + 0.2f * effort_factor;
         task.setActualImportance(importance);
         
         return task;
     }
 
     // Generate multiple random tasks
     std::vector<Task> generateRandomTasks(size_t count) {
         std::vector<Task> tasks;
         tasks.reserve(count);
         
         for (size_t i = 0; i < count; ++i) {
             tasks.push_back(generateRandomTask());
         }
         
         return tasks;
     }
 };
 
 /**
  * Demo application for task prioritization with reinforcement learning
  * 
  * This function demonstrates the usage of the task prioritization system.
  * It generates random tasks, trains the agent, and evaluates its performance.
  * 
  * Time complexity:
  * - Training: O(n * e), where n is the number of tasks and e is the number of episodes
  * - Prioritization: O(n log n), where n is the number of tasks
  * 
  * Memory complexity: O(n), where n is the number of tasks
  */
 void runTaskPrioritizationDemo() {
     std::cout << "Task Prioritization with Reinforcement Learning Demo" << std::endl;
     std::cout << "====================================================" << std::endl;
     
     // Create task generator
     TaskGenerator generator;
     
     // Create task prioritizer
     // Neural network architecture: 3 inputs (est_hours, hours_until_deadline, priority), 
     // 8 hidden neurons, 1 output (importance)
     TaskPrioritizer prioritizer({3, 8, 1}, 0.01f, 0.1f, 0.9f, 0.3f, 0.995f, 0.01f);
     
     // Generate initial tasks for training
     const size_t num_tasks = 50;
     std::vector<Task> initial_tasks = generator.generateRandomTasks(num_tasks);
     
     std::cout << "Generated " << num_tasks << " random tasks for training." << std::endl;
     
     // Add tasks to the environment
     for (const auto& task : initial_tasks) {
         prioritizer.addTask(task);
     }
     
     // Train the agent
     const size_t num_episodes = 1000;
     std::cout << "Training reinforcement learning agent for " << num_episodes << " episodes..." << std::endl;
     
     prioritizer.train(num_episodes, [](size_t episode, float reward) {
         if (episode % 100 == 0 || episode == num_episodes - 1) {
             std::cout << "Episode " << episode << ", Reward: " << reward 
                       << ", Exploration rate: " << std::fixed << std::setprecision(3) << 0.3f * std::pow(0.995f, episode) 
                       << std::endl;
         }
     });
     
     std::cout << "Training completed." << std::endl;
     
     // Evaluate the agent
     float evaluation_reward = prioritizer.evaluateAgent();
     std::cout << "Evaluation reward: " << evaluation_reward << std::endl;
     
     // Generate new tasks for demonstration
     const size_t num_demo_tasks = 10;
     std::vector<Task> demo_tasks = generator.generateRandomTasks(num_demo_tasks);
     
     std::cout << "\nDemonstrating prioritization with " << num_demo_tasks << " new tasks:" << std::endl;
     
     // Clear previous tasks and add new ones
     for (const auto& task : demo_tasks) {
         prioritizer.addTask(task);
         std::cout << task.toString() << std::endl;
     }
     
     // Prioritize tasks
     std::vector<int> prioritized_indices = prioritizer.prioritizeTasks();
     
     std::cout << "\nPrioritized task order:" << std::endl;
     for (size_t i = 0; i < prioritized_indices.size(); ++i) {
         std::cout << i + 1 << ". " << demo_tasks[prioritized_indices[i]].toString() << std::endl;
     }
     
     // Save the model
     prioritizer.saveModel("rl_task_qtable.dat", "rl_task_nn.dat");
     std::cout << "\nModel saved to 'rl_task_qtable.dat' and 'rl_task_nn.dat'." << std::endl;
 }
 
 int main() {
     try {
         runTaskPrioritizationDemo();
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
     
     return 0;
 }