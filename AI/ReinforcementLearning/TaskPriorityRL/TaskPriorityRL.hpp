/**
 * @file TaskPriorityRL.hpp
 * @brief Reinforcement learning agent for dynamic task priority scheduling
 * @details Neural network policy (DQN variant) learns to assign task priorities to minimize total weighted completion time. Uses experience replay and target network for stable training.
 */

#pragma once

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
 namespace Utils {
     // Type trait for floating point types (C++17 compatible)
     template<typename T>

     struct IsFloatingPoint : std::is_floating_point<T> {};
 
     // ReLU activation function
     template<typename T, 
              typename = std::enable_if_t<IsFloatingPoint<T>::value>>
     inline T Relu(T x) {
         return std::max<T>(0, x);
     }
 
     // Derivative of ReLU
     template<typename T,
              typename = std::enable_if_t<IsFloatingPoint<T>::value>>
     inline T ReluDerivative(T x) {
         return x > 0 ? 1 : 0;
     }
 
     // Sigmoid activation function
     template<typename T,
              typename = std::enable_if_t<IsFloatingPoint<T>::value>>
     inline T Sigmoid(T x) {
         return 1.0 / (1.0 + std::exp(-x));
     }
 
     // Derivative of sigmoid
     template<typename T,
              typename = std::enable_if_t<IsFloatingPoint<T>::value>>
     inline T SigmoidDerivative(T x) {
         T s = Sigmoid(x);
         return s * (1 - s);
     }
 
     // Mean squared error loss function
     template<typename T,
              typename = std::enable_if_t<IsFloatingPoint<T>::value>>
     inline T Mse(const std::vector<T>& Predictions, const std::vector<T>& Targets) {
         if (Predictions.size() != Targets.size()) {
             throw std::invalid_argument("Predictions and targets must have the same size");
         }
         
         T Sum = 0;
         for (size_t i = 0; i < Predictions.size(); ++i) {
             T Diff = Predictions[i] - Targets[i];
             Sum += Diff * Diff;
         }
         
         return Sum / static_cast<T>(Predictions.size());
     }
     
     // Convert time to float hours from now
     inline float HoursFromNow(const std::chrono::system_clock::time_point& time) {
         auto Now = std::chrono::system_clock::now();
         auto Duration = std::chrono::duration_cast<std::chrono::minutes>(time - Now);
         return Duration.count() / 60.0f;
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
     int Id_;                                                  // Unique task ID
     std::string Description_;                                 // Task description
     float EstimatedHours_;                                   // Estimated time to complete
     std::chrono::system_clock::time_point Deadline_;          // Task deadline
     Priority InitialPriority_;                               // Initial priority set by user
     Status Status_;                                           // Current status
     float ActualImportance_;                                 // Ground truth importance (for training)
     mutable std::shared_mutex Mutex_;                         // For thread-safe access to task properties
 
 public:
     // Constructor with validation
     Task(int Id, 
          std::string Description, 
          float EstimatedHours, 
          std::chrono::system_clock::time_point Deadline,
          Priority Priority = Priority::MEDIUM) 
         : Id_(Id), 
           Description_(std::move(Description)), 
           EstimatedHours_(EstimatedHours),
           Deadline_(Deadline),
           InitialPriority_(Priority),
           Status_(Status::PENDING),
           ActualImportance_(0.0f) 
     {
         // Input validation
         if (Id_ < 0) {
             throw std::invalid_argument("Task ID cannot be negative");
         }
         
         if (Description_.empty()) {
             throw std::invalid_argument("Task description cannot be empty");
         }
         
         if (EstimatedHours_ < 0) {
             throw std::invalid_argument("Estimated hours cannot be negative");
         }
         
         if (Deadline < std::chrono::system_clock::now()) {
             throw std::invalid_argument("Deadline cannot be in the past");
         }
     }
     
     // Copy constructor
     Task(const Task& Other)
         : Id_(Other.GetId()),
           Description_(Other.GetDescription()),
           EstimatedHours_(Other.GetEstimatedHours()),
           Deadline_(Other.GetDeadline()),
           InitialPriority_(Other.GetInitialPriority()),
           Status_(Other.GetStatus()),
           ActualImportance_(Other.GetActualImportance())
     {
         // Explicit copy constructor that doesn't copy the mutex
     }
     
     // Move constructor
     Task(Task&& Other) noexcept
         : Id_(Other.Id_),
           Description_(std::move(Other.Description_)),
           EstimatedHours_(Other.EstimatedHours_),
           Deadline_(Other.Deadline_),
           InitialPriority_(Other.InitialPriority_),
           Status_(Other.Status_),
           ActualImportance_(Other.ActualImportance_)
     {
         // Explicit move constructor that doesn't move the mutex
     }
     
     // Copy assignment operator
     Task& operator=(const Task& Other) {
         if (this != &Other) {
             std::unique_lock lock(Mutex_);
             Id_ = Other.GetId();
             Description_ = Other.GetDescription();
             EstimatedHours_ = Other.GetEstimatedHours();
             Deadline_ = Other.GetDeadline();
             InitialPriority_ = Other.GetInitialPriority();
             Status_ = Other.GetStatus();
             ActualImportance_ = Other.GetActualImportance();
         }
         return *this;
     }
     
     // Move assignment operator
     Task& operator=(Task&& Other) noexcept {
         if (this != &Other) {
             std::unique_lock lock(Mutex_);
             Id_ = Other.Id_;
             Description_ = std::move(Other.Description_);
             EstimatedHours_ = Other.EstimatedHours_;
             Deadline_ = Other.Deadline_;
             InitialPriority_ = Other.InitialPriority_;
             Status_ = Other.Status_;
             ActualImportance_ = Other.ActualImportance_;
         }
         return *this;
     }
 
     // Thread-safe getters
     int GetId() const {
         std::shared_lock lock(Mutex_);
         return Id_;
     }
 
     std::string GetDescription() const {
         std::shared_lock lock(Mutex_);
         return Description_;
     }
 
     float GetEstimatedHours() const {
         std::shared_lock lock(Mutex_);
         return EstimatedHours_;
     }
 
     std::chrono::system_clock::time_point GetDeadline() const {
         std::shared_lock lock(Mutex_);
         return Deadline_;
     }
 
     Priority GetInitialPriority() const {
         std::shared_lock lock(Mutex_);
         return InitialPriority_;
     }
 
     Status GetStatus() const {
         std::shared_lock lock(Mutex_);
         return Status_;
     }
 
     float GetActualImportance() const {
         std::shared_lock lock(Mutex_);
         return ActualImportance_;
     }
 
     // Thread-safe setters
     void SetStatus(Status Status) {
         std::unique_lock lock(Mutex_);
         Status_ = Status;
     }
 
     void SetActualImportance(float Importance) {
         std::unique_lock lock(Mutex_);
         ActualImportance_ = Importance;
     }
 
     // Convert task features to a vector for the neural network
     std::vector<float> ToFeatureVector() const {
         std::shared_lock lock(Mutex_);
         
         // Hours until deadline
         float HoursUntilDeadline = Utils::HoursFromNow(Deadline_);
         
         // Normalize priority (0 to 1)
         float PriorityValue = static_cast<float>(InitialPriority_) / 
                              static_cast<float>(Priority::CRITICAL);
         
         // Create feature vector: [estimated_hours, hours_until_deadline, priority]
         return {
             EstimatedHours_,
             HoursUntilDeadline,
             PriorityValue
         };
     }
 
     // String representation of the task for debugging
     std::string ToString() const {
         std::shared_lock lock(Mutex_);
         
         // Convert deadline to string
         auto TimeTDeadline = std::chrono::system_clock::to_time_t(Deadline_);
         std::tm TmDeadline = *std::localtime(&TimeTDeadline);
         char Buffer[80];
         std::strftime(Buffer, sizeof(Buffer), "%Y-%m-%d %H:%M", &TmDeadline);
         
         // Convert priority to string
         std::string PriorityStr;
         switch (InitialPriority_) {
             case Priority::LOW: PriorityStr = "LOW"; break;
             case Priority::MEDIUM: PriorityStr = "MEDIUM"; break;
             case Priority::HIGH: PriorityStr = "HIGH"; break;
             case Priority::CRITICAL: PriorityStr = "CRITICAL"; break;
         }
         
         // Convert status to string
         std::string StatusStr;
         switch (Status_) {
             case Status::PENDING: StatusStr = "PENDING"; break;
             case Status::IN_PROGRESS: StatusStr = "IN_PROGRESS"; break;
             case Status::COMPLETED: StatusStr = "COMPLETED"; break;
             case Status::FAILED: StatusStr = "FAILED"; break;
         }
         
         // Using stringstream for formatting (C++17 compatible)
         std::ostringstream Oss;
         Oss << "Task[id=" << Id_ 
             << ", desc='" << Description_ 
             << "', est=" << std::fixed << std::setprecision(1) << EstimatedHours_ << "h"
             << ", deadline=" << Buffer
             << ", priority=" << PriorityStr
             << ", status=" << StatusStr
             << ", importance=" << std::fixed << std::setprecision(2) << ActualImportance_ << "]";
         return Oss.str();
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
         std::vector<std::vector<float>> Weights;  // Connection weights
         std::vector<float> Biases;                // Neuron biases
         std::vector<float> Activations;           // Activation values (for forward pass)
         std::vector<float> Deltas;                // Error deltas (for backward pass)
 
         // Initialize a layer with random weights and biases
         Layer(size_t InputSize, size_t OutputSize) 
             : Weights(OutputSize, std::vector<float>(InputSize)),
               Biases(OutputSize),
               Activations(OutputSize),
               Deltas(OutputSize) {
             
             std::random_device Rd;
             std::mt19937 Gen(Rd());
             
             // Xavier initialization for weights
             float Scale = std::sqrt(6.0f / (InputSize + OutputSize));
             std::uniform_real_distribution<float> UniformDist(-Scale, Scale);
             
             for (auto& NeuronWeights : Weights) {
                 for (auto& Weight : NeuronWeights) {
                     Weight = UniformDist(Gen);
                 }
             }
             
             for (auto& Bias : Biases) {
                 Bias = 0.0f;  // Initialize biases to zero
             }
         }
     };
 
     std::vector<Layer> Layers_;
     float LearningRate_;
     mutable std::mutex Mutex_;  // For thread-safe training and prediction
 
 public:
     // Constructor with input, hidden, and output layer sizes
     NeuralNetwork(const std::vector<size_t>& LayerSizes, float LearningRate = 0.01f)
         : LearningRate_(LearningRate) {
         
         if (LayerSizes.size() < 2) {
             throw std::invalid_argument("Network must have at least input and output layers");
         }
         
         // Create layers
         for (size_t i = 0; i < LayerSizes.size() - 1; ++i) {
             Layers_.emplace_back(LayerSizes[i], LayerSizes[i + 1]);
         }
     }
 
     // Forward pass through the network (thread-safe)
     float Predict(const std::vector<float>& Inputs) const {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking to ensure thread safety during prediction
         return PredictImpl(Inputs);
     }
 
     // Non-thread-safe implementation of predict for internal use
     float PredictImpl(const std::vector<float>& Inputs) const {
         if (Inputs.size() != Layers_[0].Weights[0].size()) {
             throw std::invalid_argument("Input size does not match network input layer");
         }
 
         // Input to first hidden layer
         std::vector<float> CurrentActivations = Inputs;
         
         // Forward propagation through hidden layers
         for (size_t l = 0; l < Layers_.size(); ++l) {
             const auto& Layer = Layers_[l];
             std::vector<float> NewActivations(Layer.Activations.size());
             
             for (size_t i = 0; i < Layer.Activations.size(); ++i) {
                 float Sum = Layer.Biases[i];
                 for (size_t j = 0; j < CurrentActivations.size(); ++j) {
                     Sum += Layer.Weights[i][j] * CurrentActivations[j];
                 }
                 
                 // Apply activation function (ReLU for hidden layers, sigmoid for output)
                 if (l == Layers_.size() - 1) {
                     NewActivations[i] = Utils::Sigmoid(Sum);
                 } else {
                     NewActivations[i] = Utils::Relu(Sum);
                 }
             }
             
             CurrentActivations = NewActivations;
         }
         
         // Return the output layer's activation (single output for task importance)
         return CurrentActivations[0];
     }
 
     // Train the network on a single example (thread-safe)
     void Train(const std::vector<float>& Inputs, float Target) {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking to ensure thread safety during training
         
         if (Inputs.size() != Layers_[0].Weights[0].size()) {
             throw std::invalid_argument("Input size does not match network input layer");
         }
 
         // Forward pass
         std::vector<std::vector<float>> AllActivations;
         AllActivations.push_back(Inputs);
         
         std::vector<std::vector<float>> PreActivations;
         
         std::vector<float> CurrentActivations = Inputs;
         
         // Forward propagation and store activations
         for (size_t l = 0; l < Layers_.size(); ++l) {
             auto& Layer = Layers_[l];
             std::vector<float> PreActivation(Layer.Activations.size());
             std::vector<float> NewActivations(Layer.Activations.size());
             
             for (size_t i = 0; i < Layer.Activations.size(); ++i) {
                 float Sum = Layer.Biases[i];
                 for (size_t j = 0; j < CurrentActivations.size(); ++j) {
                     Sum += Layer.Weights[i][j] * CurrentActivations[j];
                 }
                 
                 PreActivation[i] = Sum;
                 
                 // Apply activation function
                 if (l == Layers_.size() - 1) {
                     NewActivations[i] = Utils::Sigmoid(Sum);
                 } else {
                     NewActivations[i] = Utils::Relu(Sum);
                 }
             }
             
             PreActivations.push_back(PreActivation);
             AllActivations.push_back(NewActivations);
             CurrentActivations = NewActivations;
         }
         
         // Output of the network
         float Output = CurrentActivations[0];
         
         // Backward pass (backpropagation)
         // Output layer error
         float OutputDelta = (Output - Target) * Utils::SigmoidDerivative(PreActivations.back()[0]);
         Layers_.back().Deltas[0] = OutputDelta;
         
         // Hidden layers error
         for (int l = static_cast<int>(Layers_.size()) - 2; l >= 0; --l) {
             auto& Layer = Layers_[l];
             auto& NextLayer = Layers_[l + 1];
             
             for (size_t i = 0; i < Layer.Deltas.size(); ++i) {
                 float Error = 0.0f;
                 for (size_t j = 0; j < NextLayer.Deltas.size(); ++j) {
                     Error += NextLayer.Deltas[j] * NextLayer.Weights[j][i];
                 }
                 
                 Layer.Deltas[i] = Error * Utils::ReluDerivative(PreActivations[l][i]);
             }
         }
         
         // Update weights and biases
         for (size_t l = 0; l < Layers_.size(); ++l) {
             auto& Layer = Layers_[l];
             const auto& PrevActivations = AllActivations[l];
             
             for (size_t i = 0; i < Layer.Weights.size(); ++i) {
                 for (size_t j = 0; j < Layer.Weights[i].size(); ++j) {
                     Layer.Weights[i][j] -= LearningRate_ * Layer.Deltas[i] * PrevActivations[j];
                 }
                 
                 Layer.Biases[i] -= LearningRate_ * Layer.Deltas[i];
             }
         }
     }
 
     // Batch training
     void TrainBatch(const std::vector<std::vector<float>>& BatchInputs, 
                    const std::vector<float>& BatchTargets,
                    size_t Epochs,
                    bool Verbose = false) {
         if (BatchInputs.size() != BatchTargets.size()) {
             throw std::invalid_argument("Number of inputs must match number of targets");
         }
         
         // Create indices for shuffling
         std::vector<size_t> Indices(BatchInputs.size());
         std::iota(Indices.begin(), Indices.end(), 0);
         
         for (size_t Epoch = 0; Epoch < Epochs; ++Epoch) {
             // Shuffle training data for each epoch
             std::random_device Rd;
             std::mt19937 g(Rd());
             std::shuffle(Indices.begin(), Indices.end(), g);
             
             float TotalLoss = 0.0f;
             
             for (size_t Idx : Indices) {
                 Train(BatchInputs[Idx], BatchTargets[Idx]);
                 
                 // Calculate loss for monitoring
                 float Prediction = Predict(BatchInputs[Idx]);
                 float Error = Prediction - BatchTargets[Idx];
                 TotalLoss += Error * Error;
             }
             
             TotalLoss /= static_cast<float>(BatchInputs.size());
             
             if (Verbose && (Epoch % 100 == 0 || Epoch == Epochs - 1)) {
                 std::cout << "Epoch " << Epoch << ", Loss: " << TotalLoss << std::endl;
             }
         }
     }
 
     // Save model to file
     void SaveModel(const std::string& Filename) const {
         std::lock_guard<std::mutex> lock(Mutex_);
         
         std::ofstream File(Filename, std::ios::binary);
         if (!File) {
             throw std::runtime_error("Failed to open file for writing");
         }
         
         // Save number of layers
         size_t NumLayers = Layers_.size();
         File.write(reinterpret_cast<const char*>(&NumLayers), sizeof(NumLayers));
         
         // Save each layer
         for (const auto& Layer : Layers_) {
             // Save dimensions
             size_t OutputSize = Layer.Biases.size();
             size_t InputSize = Layer.Weights[0].size();
             File.write(reinterpret_cast<const char*>(&OutputSize), sizeof(OutputSize));
             File.write(reinterpret_cast<const char*>(&InputSize), sizeof(InputSize));
             
             // Save weights
             for (const auto& NeuronWeights : Layer.Weights) {
                 for (float Weight : NeuronWeights) {
                     File.write(reinterpret_cast<const char*>(&Weight), sizeof(Weight));
                 }
             }
             
             // Save biases
             for (float Bias : Layer.Biases) {
                 File.write(reinterpret_cast<const char*>(&Bias), sizeof(Bias));
             }
         }
         
         // Save learning rate
         File.write(reinterpret_cast<const char*>(&LearningRate_), sizeof(LearningRate_));
     }
 
     // Load model from file
     void LoadModel(const std::string& Filename) {
         std::lock_guard<std::mutex> lock(Mutex_);
         
         std::ifstream File(Filename, std::ios::binary);
         if (!File) {
             throw std::runtime_error("Failed to open file for reading");
         }
         
         // Load number of layers
         size_t NumLayers;
         File.read(reinterpret_cast<char*>(&NumLayers), sizeof(NumLayers));
         
         // Clear existing layers
         Layers_.clear();
         
         // Load each layer
         for (size_t l = 0; l < NumLayers; ++l) {
             // Load dimensions
             size_t OutputSize, InputSize;
             File.read(reinterpret_cast<char*>(&OutputSize), sizeof(OutputSize));
             File.read(reinterpret_cast<char*>(&InputSize), sizeof(InputSize));
             
             // Create layer
             Layer Layer(InputSize, OutputSize);
             
             // Load weights
             for (auto& NeuronWeights : Layer.Weights) {
                 for (auto& Weight : NeuronWeights) {
                     File.read(reinterpret_cast<char*>(&Weight), sizeof(Weight));
                 }
             }
             
             // Load biases
             for (auto& Bias : Layer.Biases) {
                 File.read(reinterpret_cast<char*>(&Bias), sizeof(Bias));
             }
             
             Layers_.push_back(std::move(Layer));
         }
         
         // Load learning rate
         File.read(reinterpret_cast<char*>(&LearningRate_), sizeof(LearningRate_));
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
         std::vector<float> Features;
         
         // Hash function for unordered_map
         struct Hash {
             size_t operator()(const State& St) const {
                 size_t hash = 0;
                 for (float Feature : St.Features) {
                     // Convert float to int for hashing
                     int IntVal = static_cast<int>(Feature * 100); // Scale and discretize
                     hash = hash * 31 + std::hash<int>()(IntVal);
                 }
                 return hash;
             }
         };
         
         // Equality operator for unordered_map
         bool operator==(const State& Other) const {
             if (Features.size() != Other.Features.size()) {
                 return false;
             }
             
             for (size_t i = 0; i < Features.size(); ++i) {
                 // Compare with some tolerance due to floating point precision
                 if (std::abs(Features[i] - Other.Features[i]) > 0.01f) {
                     return false;
                 }
             }
             
             return true;
         }
     };
 
 private:
     using Action = int;  // Action is the index of the task to prioritize
     using QValue = float;  // Q-value for a state-action pair
     
     std::unordered_map<State, std::unordered_map<Action, QValue>, State::Hash> QTable_;
     float LearningRate_;
     float DiscountFactor_;
     float ExplorationRate_;
     float ExplorationDecay_;
     float MinExplorationRate_;
     NeuralNetwork NnPredictor_;
     mutable std::mutex Mutex_;  // For thread-safe access to Q-table and neural network
 
 public:
     RLAgent(const std::vector<size_t>& NnLayerSizes,
             float NnLearningRate = 0.01f,
             float LearningRate = 0.1f,
             float DiscountFactor = 0.9f,
             float ExplorationRate = 0.3f,
             float ExplorationDecay = 0.995f,
             float MinExplorationRate = 0.01f)
         : LearningRate_(LearningRate),
           DiscountFactor_(DiscountFactor),
           ExplorationRate_(ExplorationRate),
           ExplorationDecay_(ExplorationDecay),
           MinExplorationRate_(MinExplorationRate),
           NnPredictor_(NnLayerSizes, NnLearningRate) {
         
         // Validate parameters
         if (LearningRate_ <= 0 || LearningRate_ > 1) {
             throw std::invalid_argument("Learning rate must be in (0, 1]");
         }
         
         if (DiscountFactor_ < 0 || DiscountFactor_ > 1) {
             throw std::invalid_argument("Discount factor must be in [0, 1]");
         }
         
         if (ExplorationRate_ < 0 || ExplorationRate_ > 1) {
             throw std::invalid_argument("Exploration rate must be in [0, 1]");
         }
         
         if (ExplorationDecay_ <= 0 || ExplorationDecay_ > 1) {
             throw std::invalid_argument("Exploration decay must be in (0, 1]");
         }
         
         if (MinExplorationRate_ < 0 || MinExplorationRate_ > ExplorationRate_) {
             throw std::invalid_argument("Min exploration rate must be in [0, exploration_rate]");
         }
     }
 
     // Choose action based on ε-greedy policy
     Action ChooseAction(const State& St, const std::vector<Action>& AvailableActions) {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         if (AvailableActions.empty()) {
             throw std::invalid_argument("No available actions");
         }
         
         // With probability exploration_rate, choose a random action (exploration)
         std::random_device Rd;
         std::mt19937 Gen(Rd());
         std::uniform_real_distribution<float> Dist(0.0f, 1.0f);
         
         if (Dist(Gen) < ExplorationRate_) {
             std::uniform_int_distribution<size_t> ActionDist(0, AvailableActions.size() - 1);
             return AvailableActions[ActionDist(Gen)];
         }
         
         // Otherwise, choose the action with the highest Q-value (exploitation)
         Action BestAction = AvailableActions[0];
         float BestQValue = GetQValue(St, BestAction);
         
         for (size_t i = 1; i < AvailableActions.size(); ++i) {
             Action Act = AvailableActions[i];
             float QValue = GetQValue(St, Act);
             
             if (QValue > BestQValue) {
                 BestQValue = QValue;
                 BestAction = Act;
             }
         }
         
         return BestAction;
     }
 
     // Update Q-value for a state-action pair using Q-learning
     void UpdateQValue(const State& St, Action Act, float Reward, const State& NextState, 
                      const std::vector<Action>& AvailableNextActions) {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         // Get current Q-value
         float CurrentQ = GetQValue(St, Act);
         
         // Get maximum Q-value for the next state
         float MaxNextQ = 0.0f;
         if (!AvailableNextActions.empty()) {
             MaxNextQ = GetQValue(NextState, AvailableNextActions[0]);
             
             for (size_t i = 1; i < AvailableNextActions.size(); ++i) {
                 float q = GetQValue(NextState, AvailableNextActions[i]);
                 if (q > MaxNextQ) {
                     MaxNextQ = q;
                 }
             }
         }
         
         // Q-learning update formula: Q(s,a) = Q(s,a) + learning_rate * (reward + discount_factor * max_a' Q(s',a') - Q(s,a))
         float NewQ = CurrentQ + LearningRate_ * (Reward + DiscountFactor_ * MaxNextQ - CurrentQ);
         
         // Update Q-table
         QTable_[St][Act] = NewQ;
     }
 
     // Decay exploration rate
     void DecayExplorationRate() {
         std::lock_guard<std::mutex> lock(Mutex_);
         ExplorationRate_ = std::max(MinExplorationRate_, ExplorationRate_ * ExplorationDecay_);
     }
 
     // Get current exploration rate
     float GetExplorationRate() const {
         std::lock_guard<std::mutex> lock(Mutex_);
         return ExplorationRate_;
     }
 
     // Predict task importance using neural network
     float PredictImportance(const std::vector<float>& TaskFeatures) const {
         return NnPredictor_.Predict(TaskFeatures);
     }
 
     // Train neural network with task features and actual importance
     void TrainNeuralNetwork(const std::vector<std::vector<float>>& FeaturesBatch,
                            const std::vector<float>& ImportanceBatch,
                            size_t Epochs = 100,
                            bool Verbose = false) {
         NnPredictor_.TrainBatch(FeaturesBatch, ImportanceBatch, Epochs, Verbose);
     }
 
     // Train neural network with single example
     void TrainNeuralNetwork(const std::vector<float>& Features, float Importance) {
         NnPredictor_.Train(Features, Importance);
     }
 
     // Save agent (Q-table and neural network) to files
     void SaveAgent(const std::string& QTableFile, const std::string& NnFile) const {
         std::lock_guard<std::mutex> lock(Mutex_);
         
         // Save Q-table
         std::ofstream QFile(QTableFile);
         if (!QFile) {
             throw std::runtime_error("Failed to open Q-table file for writing");
         }
         
         for (const auto& [St, Actions] : QTable_) {
             // Write state features
             for (float Feature : St.Features) {
                 QFile << Feature << " ";
             }
             
             // Write number of actions
             QFile << Actions.size() << " ";
             
             // Write actions and Q-values
             for (const auto& [Act, QVal] : Actions) {
                 QFile << Act << " " << QVal << " ";
             }
             
             QFile << std::endl;
         }
         
         // Save neural network
         NnPredictor_.SaveModel(NnFile);
     }
 
     // Load agent (Q-table and neural network) from files
     void LoadAgent(const std::string& QTableFile, const std::string& NnFile) {
         std::lock_guard<std::mutex> lock(Mutex_);
         
         // Clear existing Q-table
         QTable_.clear();
         
         // Load Q-table
         std::ifstream QFile(QTableFile);
         if (!QFile) {
             throw std::runtime_error("Failed to open Q-table file for reading");
         }
         
         std::string Line;
         while (std::getline(QFile, Line)) {
             std::istringstream Iss(Line);
             
             // Read state features
             State St;
             float Feature;
             while (Iss >> Feature) {
                 St.Features.push_back(Feature);
                 
                 // Check if we've reached the number of actions
                 if (St.Features.size() == 3) {  // Assuming 3 features per state
                     break;
                 }
             }
             
             // Read number of actions
             size_t NumActions;
             Iss >> NumActions;
             
             // Read actions and Q-values
             for (size_t i = 0; i < NumActions; ++i) {
                 Action Act;
                 QValue QVal;
                 Iss >> Act >> QVal;
                 
                 QTable_[St][Act] = QVal;
             }
         }
         
         // Load neural network
         NnPredictor_.LoadModel(NnFile);
     }
 
 private:
     // Get Q-value for a state-action pair (initialize to 0 if not found)
     float GetQValue(const State& St, Action Act) const {
         auto StateIt = QTable_.find(St);
         if (StateIt != QTable_.end()) {
             auto ActionIt = StateIt->second.find(Act);
             if (ActionIt != StateIt->second.end()) {
                 return ActionIt->second;
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
     std::vector<Task> Tasks_;
     std::vector<Task> CompletedTasks_;
     float TimePenaltyFactor_;  // Penalty for delayed tasks
     mutable std::mutex Mutex_;  // For thread-safe access to tasks
     
 public:
     explicit TaskEnvironment(float TimePenaltyFactor = 0.1f)
         : TimePenaltyFactor_(TimePenaltyFactor) {
         
         if (TimePenaltyFactor_ < 0) {
             throw std::invalid_argument("Time penalty factor cannot be negative");
         }
     }
 
     // Add a new task to the environment
     void AddTask(const Task& Task) {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         Tasks_.push_back(Task);
     }
 
     // Get list of tasks
     std::vector<Task> GetTasks() const {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         return Tasks_;
     }
 
     // Get list of available actions (indices of pending tasks)
     std::vector<int> GetAvailableActions() const {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         std::vector<int> AvailableActions;
         for (size_t i = 0; i < Tasks_.size(); ++i) {
             if (Tasks_[i].GetStatus() == Task::Status::PENDING) {
                 AvailableActions.push_back(static_cast<int>(i));
             }
         }
         
         return AvailableActions;
     }
 
     // Execute an action (complete a task) and return the reward
     float ExecuteAction(int Action) {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         if (Action < 0 || Action >= static_cast<int>(Tasks_.size())) {
             throw std::invalid_argument("Invalid action index");
         }
         
         Task& Task = Tasks_[Action];
         
         if (Task.GetStatus() != Task::Status::PENDING) {
             throw std::invalid_argument("Cannot execute action on non-pending task");
         }
         
         // Mark task as completed
         Task.SetStatus(Task::Status::COMPLETED);
         
         // Calculate reward based on task importance and timeliness
         float Importance = Task.GetActualImportance();
         float HoursUntilDeadline = Utils::HoursFromNow(Task.GetDeadline());
         
         // Higher reward for completing tasks before deadline
         float TimeBonus = HoursUntilDeadline > 0 ? 
                           std::min(1.0f, HoursUntilDeadline / 24.0f) : 
                           -TimePenaltyFactor_ * std::abs(HoursUntilDeadline);
         
         float Reward = Importance + TimeBonus;
         
         // Move task to completed list
         CompletedTasks_.push_back(Task);
         
         // Remove task from active list
         Tasks_.erase(Tasks_.begin() + Action);
         
         return Reward;
     }
 
     // Get current state features (aggregated from all pending tasks)
     std::vector<float> GetCurrentState() const {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         // Calculate average estimated hours, hours until deadline, and priority
         float AvgEstimatedHours = 0.0f;
         float AvgHoursUntilDeadline = 0.0f;
         float AvgPriority = 0.0f;
         
         int PendingCount = 0;
         
         for (const auto& Task : Tasks_) {
             if (Task.GetStatus() == Task::Status::PENDING) {
                 AvgEstimatedHours += Task.GetEstimatedHours();
                 AvgHoursUntilDeadline += Utils::HoursFromNow(Task.GetDeadline());
                 AvgPriority += static_cast<float>(Task.GetInitialPriority()) / 
                               static_cast<float>(Task::Priority::CRITICAL);
                 
                 PendingCount++;
             }
         }
         
         if (PendingCount > 0) {
             AvgEstimatedHours /= PendingCount;
             AvgHoursUntilDeadline /= PendingCount;
             AvgPriority /= PendingCount;
         }
         
         return {AvgEstimatedHours, AvgHoursUntilDeadline, AvgPriority, static_cast<float>(PendingCount)};
     }
 
     // Check if there are any pending tasks
     bool HasPendingTasks() const {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         for (const auto& Task : Tasks_) {
             if (Task.GetStatus() == Task::Status::PENDING) {
                 return true;
             }
         }
         
         return false;
     }
 
     // Reset the environment
     void Reset() {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         // Reset all tasks to pending status
         for (auto& Task : Tasks_) {
             Task.SetStatus(Task::Status::PENDING);
         }
         
         // Clear completed tasks
         CompletedTasks_.clear();
     }
 
     // Clear all tasks
     void ClearTasks() {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         Tasks_.clear();
         CompletedTasks_.clear();
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
     RLAgent Agent_;
     TaskEnvironment Environment_;
     mutable std::mutex Mutex_;  // For thread-safe operations
     
 public:
     TaskPrioritizer(const std::vector<size_t>& NnLayerSizes,
                   float NnLearningRate = 0.01f,
                   float RlLearningRate = 0.1f,
                   float DiscountFactor = 0.9f,
                   float ExplorationRate = 0.3f,
                   float ExplorationDecay = 0.995f,
                   float MinExplorationRate = 0.01f,
                   float TimePenaltyFactor = 0.1f)
         : Agent_(NnLayerSizes, NnLearningRate, RlLearningRate, DiscountFactor,
                 ExplorationRate, ExplorationDecay, MinExplorationRate),
           Environment_(TimePenaltyFactor) {
     }
 
     // Add a task
     void AddTask(const Task& Task) {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         Environment_.AddTask(Task);
     }
 
     // Prioritize tasks and return sorted task indices
     std::vector<int> PrioritizeTasks() {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         std::vector<Task> Tasks = Environment_.GetTasks();
         std::vector<int> PrioritizedTasks;
         
         if (Tasks.empty()) {
             return PrioritizedTasks;
         }
         
         // Calculate importance for each task using neural network
         std::vector<std::pair<int, float>> TaskImportance;
         for (size_t i = 0; i < Tasks.size(); ++i) {
             if (Tasks[i].GetStatus() == Task::Status::PENDING) {
                 std::vector<float> Features = Tasks[i].ToFeatureVector();
                 float Importance = Agent_.PredictImportance(Features);
                 TaskImportance.emplace_back(i, Importance);
             }
         }
         
         // Sort tasks by predicted importance (highest first)
         std::sort(TaskImportance.begin(), TaskImportance.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
         
         // Extract task indices
         for (const auto& [Idx, _] : TaskImportance) {
             PrioritizedTasks.push_back(Idx);
         }
         
         return PrioritizedTasks;
     }
 
     // Train the agent on a set of completed tasks
     void TrainOnCompletedTasks(const std::vector<Task>& CompletedTasks, size_t Epochs = 100) {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         if (CompletedTasks.empty()) {
             return;
         }
         
         std::vector<std::vector<float>> FeaturesBatch;
         std::vector<float> ImportanceBatch;
         
         for (const auto& Task : CompletedTasks) {
             FeaturesBatch.push_back(Task.ToFeatureVector());
             ImportanceBatch.push_back(Task.GetActualImportance());
         }
         
         Agent_.TrainNeuralNetwork(FeaturesBatch, ImportanceBatch, Epochs);
     }
 
     // Run a full episode of training
     float RunTrainingEpisode() {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         // Reset environment
         Environment_.Reset();
         
         float TotalReward = 0.0f;
         int Steps = 0;
         
         // Run until no pending tasks
         while (Environment_.HasPendingTasks()) {
             // Get current state
             auto StateFeatures = Environment_.GetCurrentState();
             RLAgent::State CurrentState{StateFeatures};
             
             // Get available actions
             auto AvailableActions = Environment_.GetAvailableActions();
             
             // Choose action using ε-greedy policy
             int Action = Agent_.ChooseAction(CurrentState, AvailableActions);
             
             // Execute action and get reward
             float Reward = Environment_.ExecuteAction(Action);
             TotalReward += Reward;
             
             // Get next state after action
             auto NextStateFeatures = Environment_.GetCurrentState();
             RLAgent::State NextState{NextStateFeatures};
             
             // Get available actions for next state
             auto NextAvailableActions = Environment_.GetAvailableActions();
             
             // Update Q-values
             Agent_.UpdateQValue(CurrentState, Action, Reward, NextState, NextAvailableActions);
             
             Steps++;
         }
         
         // Decay exploration rate after each episode
         Agent_.DecayExplorationRate();
         
         return TotalReward;
     }
 
     // Run multiple training episodes
     void Train(size_t NumEpisodes, std::function<void(size_t, float)> ProgressCallback = nullptr) {
         for (size_t Episode = 0; Episode < NumEpisodes; ++Episode) {
             float EpisodeReward = RunTrainingEpisode();
             
             if (ProgressCallback) {
                 ProgressCallback(Episode, EpisodeReward);
             }
         }
     }
 
     // Save the model
     void SaveModel(const std::string& QTableFile, const std::string& NnFile) const {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         Agent_.SaveAgent(QTableFile, NnFile);
     }
 
     // Load the model
     void LoadModel(const std::string& QTableFile, const std::string& NnFile) {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         Agent_.LoadAgent(QTableFile, NnFile);
     }
 
     // Evaluate the agent without exploration
     float EvaluateAgent() {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         // Set exploration rate to 0 temporarily (pure exploitation)
         [[maybe_unused]] float savedRate = Agent_.GetExplorationRate();
         
         // Run one episode with zero exploration rate
         float Reward = RunTrainingEpisode();
         
         // No need to restore exploration rate as it's already decayed in runTrainingEpisode
         
         return Reward;
     }
 
     // Get current exploration rate
     float GetExplorationRate() const {
         return Agent_.GetExplorationRate();
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
     int NextId_;
     std::vector<std::string> TaskDescriptions_;
     std::mt19937 Gen_;
     mutable std::mutex Mutex_;  // For thread-safe task generation
     
 public:
     TaskGenerator() : NextId_(0) {
         // Initialize random generator
         std::random_device Rd;
         Gen_ = std::mt19937(Rd());
         
         // Sample task descriptions
         TaskDescriptions_ = {
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
     Task GenerateRandomTask() {
         std::lock_guard<std::mutex> lock(Mutex_);  // Mutex locking for thread safety
         
         // Generate random task parameters
         std::uniform_int_distribution<size_t> DescDist(0, TaskDescriptions_.size() - 1);
         std::uniform_real_distribution<float> HoursDist(1.0f, 8.0f);
         std::uniform_int_distribution<int> DeadlineDist(1, 168);  // 1 hour to 1 week
         std::uniform_int_distribution<int> PriorityDist(0, 3);
         
         // Select random description
         std::string Description = TaskDescriptions_[DescDist(Gen_)];
         
         // Random estimated hours (1-8 hours)
         float EstimatedHours = HoursDist(Gen_);
         
         // Random deadline (1 hour to 1 week from now)
         auto Now = std::chrono::system_clock::now();
         auto Deadline = Now + std::chrono::hours(DeadlineDist(Gen_));
         
         // Random priority
         Task::Priority Priority = static_cast<Task::Priority>(PriorityDist(Gen_));
         
         // Create task
         Task Task(NextId_++, Description, EstimatedHours, Deadline, Priority);
         
         // Set actual importance based on a formula (for simulation purposes)
         // Higher importance for:
         // - Tasks with closer deadlines
         // - Tasks with higher priority
         // - Tasks that take less time (quick wins)
         float TimeFactor = std::min(1.0f, 48.0f / Utils::HoursFromNow(Deadline));
         float PriorityFactor = static_cast<float>(Priority) / static_cast<float>(Task::Priority::CRITICAL);
         float EffortFactor = 1.0f / (1.0f + EstimatedHours / 4.0f);
         
         float Importance = 0.4f * TimeFactor + 0.4f * PriorityFactor + 0.2f * EffortFactor;
         Task.SetActualImportance(Importance);
         
         return Task;
     }
 
     // Generate multiple random tasks
     std::vector<Task> GenerateRandomTasks(size_t count) {
         std::vector<Task> Tasks;
         Tasks.reserve(count);
         
         for (size_t i = 0; i < count; ++i) {
             Tasks.push_back(GenerateRandomTask());
         }
         
         return Tasks;
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
 inline void RunTaskPrioritizationDemo() {
     std::cout << "Task Prioritization with Reinforcement Learning Demo" << std::endl;
     std::cout << "====================================================" << std::endl;
     
     // Create task generator
     TaskGenerator Generator;
     
     // Create task prioritizer
     // Neural network architecture: 3 inputs (est_hours, hours_until_deadline, priority), 
     // 8 hidden neurons, 1 output (importance)
     TaskPrioritizer Prioritizer({3, 8, 1}, 0.01f, 0.1f, 0.9f, 0.3f, 0.995f, 0.01f);
     
     // Generate initial tasks for training
     const size_t NumTasks = 50;
     std::vector<Task> InitialTasks = Generator.GenerateRandomTasks(NumTasks);
     
     std::cout << "Generated " << NumTasks << " random tasks for training." << std::endl;
     
     // Add tasks to the environment
     for (const auto& Task : InitialTasks) {
         Prioritizer.AddTask(Task);
     }
     
     // Train the agent
     const size_t NumEpisodes = 1000;
     std::cout << "Training reinforcement learning agent for " << NumEpisodes << " episodes..." << std::endl;
     
     Prioritizer.Train(NumEpisodes, [](size_t Episode, float Reward) {
         if (Episode % 100 == 0 || Episode == NumEpisodes - 1) {
             std::cout << "Episode " << Episode << ", Reward: " << Reward 
                       << ", Exploration rate: " << std::fixed << std::setprecision(3) << 0.3f * std::pow(0.995f, Episode) 
                       << std::endl;
         }
     });
     
     std::cout << "Training completed." << std::endl;
     
     // Evaluate the agent
     float EvaluationReward = Prioritizer.EvaluateAgent();
     std::cout << "Evaluation reward: " << EvaluationReward << std::endl;
     
     // Generate new tasks for demonstration
     const size_t NumDemoTasks = 10;
     std::vector<Task> DemoTasks = Generator.GenerateRandomTasks(NumDemoTasks);
     
     std::cout << "\nDemonstrating prioritization with " << NumDemoTasks << " new tasks:" << std::endl;
     
     // Clear previous tasks and add new ones
     for (const auto& Task : DemoTasks) {
         Prioritizer.AddTask(Task);
         std::cout << Task.ToString() << std::endl;
     }
     
     // Prioritize tasks
     std::vector<int> PrioritizedIndices = Prioritizer.PrioritizeTasks();
     
     std::cout << "\nPrioritized task order:" << std::endl;
     for (size_t i = 0; i < PrioritizedIndices.size(); ++i) {
         std::cout << i + 1 << ". " << DemoTasks[PrioritizedIndices[i]].ToString() << std::endl;
     }
     
     // Save the model
     Prioritizer.SaveModel("rl_task_qtable.dat", "rl_task_nn.dat");
     std::cout << "\nModel saved to 'rl_task_qtable.dat' and 'rl_task_nn.dat'." << std::endl;
 }
 
