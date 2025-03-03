/**
 * @file HealthPredictionNN.hpp
 * @brief Neural network for health outcome prediction
 * @details Multi-layer feedforward neural network with backpropagation trained on health diagnostic data. Includes data normalization, training loop, and accuracy evaluation.
 */

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <iomanip>

/**
 * @brief A Neural Network implementation for health score prediction
 * 
 * This implementation uses a feedforward neural network with configurable layers
 * to predict a health score based on multiple health-related features.
 * 
 * The network supports:
 * - Multiple hidden layers with configurable neurons
 * - Various activation functions
 * - Backpropagation for training
 * - Mini-batch gradient descent
 * - Early stopping
 * - Feature normalization
 * - Thread-safe inference
 */

class HealthScorePredictor {
public:
    // Activation function types
    enum class ActivationType {
        SIGMOID,
        RELU,
        TANH,
        LINEAR
    };
    
    // Feature definition with normalization parameters
    struct Feature {
        std::string Name;
        double MinValue;
        double MaxValue;
        double Mean;
        double StdDev;
        
        // Normalize a value using min-max scaling
        double Normalize(double value) const {
            if (StdDev > 0) {
                // Z-score normalization
                return (value - Mean) / StdDev;
            } else {
                // Min-max normalization as fallback
                return (value - MinValue) / (MaxValue - MinValue) * 2.0 - 1.0;
            }
        }
        
        // Denormalize a value
        double Denormalize(double NormalizedValue) const {
            if (StdDev > 0) {
                // Z-score denormalization
                return NormalizedValue * StdDev + Mean;
            } else {
                // Min-max denormalization
                return (NormalizedValue + 1.0) / 2.0 * (MaxValue - MinValue) + MinValue;
            }
        }
    };
    
private:
    // Neural network architecture
    struct Neuron;
    struct Layer;
    
    struct Connection {
        double Weight;
        double DeltaWeight;
        
        Connection() : Weight(0.0), DeltaWeight(0.0) {}
    };
    
    struct Neuron {
        std::vector<Connection> OutputWeights;
        double OutputValue;
        double Gradient;
        unsigned Index;
        
        Neuron(unsigned NumOutputs, unsigned Idx) : OutputValue(0.0), Gradient(0.0), Index(Idx) {
            // Initialize with random weights
            static std::random_device Rd;
            static std::mt19937 Gen(Rd());
            static std::normal_distribution<double> Dist(0.0, 1.0);
            
            OutputWeights.resize(NumOutputs);
            for (auto& Connection : OutputWeights) {
                // Xavier/Glorot initialization
                Connection.Weight = Dist(Gen) * sqrt(2.0 / (NumOutputs + 1));
            }
        }
    };
    
    struct Layer {
        std::vector<Neuron> Neurons;
        ActivationType Activation;
        
        Layer() = default;
        
        Layer(unsigned NumNeurons, unsigned NumOutputs, ActivationType Act)
            : Activation(Act) {
            Neurons.reserve(NumNeurons + 1); // +1 for bias neuron
            
            // Create neurons in the layer
            for (unsigned i = 0; i < NumNeurons; ++i) {
                Neurons.emplace_back(NumOutputs, i);
            }
            
            // Add bias neuron (with constant output of 1.0)
            Neurons.emplace_back(NumOutputs, NumNeurons);
            Neurons.back().OutputValue = 1.0;
        }
    };
    
    std::vector<Layer> Layers;
    std::vector<Feature> Features;
    double LearningRate;
    double Momentum;
    double Error;
    unsigned Epochs;
    mutable std::mutex InferenceMutex; // For thread-safe inference, mutable to allow const methods to lock it
    
    // Activation functions
    double Activate(double x, ActivationType Type) const {
        switch (Type) {
            case ActivationType::SIGMOID:
                return 1.0 / (1.0 + exp(-x));
            case ActivationType::RELU:
                return std::max(0.0, x);
            case ActivationType::TANH:
                return tanh(x);
            case ActivationType::LINEAR:
            default:
                return x;
        }
    }
    
    // Derivatives of activation functions
    double ActivateDerivative(double x, ActivationType Type) const {
        switch (Type) {
            case ActivationType::SIGMOID:
                return x * (1.0 - x); // x is already the sigmoid output
            case ActivationType::RELU:
                return x > 0.0 ? 1.0 : 0.0;
            case ActivationType::TANH:
                return 1.0 - x * x; // x is already the tanh output
            case ActivationType::LINEAR:
            default:
                return 1.0;
        }
    }
    
    // Forward propagation (inference)
    void FeedForward(const std::vector<double>& InputValues) {
        // Set input layer values
        for (unsigned i = 0; i < InputValues.size(); ++i) {
            Layers[0].Neurons[i].OutputValue = InputValues[i];
        }
        
        // Forward propagate through all subsequent layers
        for (unsigned LayerIdx = 1; LayerIdx < Layers.size(); ++LayerIdx) {
            auto& PrevLayer = Layers[LayerIdx - 1];
            auto& CurrentLayer = Layers[LayerIdx];
            
            // Process all neurons except the bias neuron
            for (unsigned n = 0; n < CurrentLayer.Neurons.size() - 1; ++n) {
                double Sum = 0.0;
                
                // Sum weighted inputs from previous layer
                for (unsigned PrevN = 0; PrevN < PrevLayer.Neurons.size(); ++PrevN) {
                    Sum += PrevLayer.Neurons[PrevN].OutputValue * 
                           PrevLayer.Neurons[PrevN].OutputWeights[n].Weight;
                }
                
                // Apply activation function
                CurrentLayer.Neurons[n].OutputValue = 
                    Activate(Sum, CurrentLayer.Activation);
            }
        }
    }
    
    // Backpropagation
    void BackPropagate(const std::vector<double>& TargetValues) {
        // Calculate output layer gradients
        auto& OutputLayer = Layers.back();
        Error = 0.0;
        
        for (unsigned n = 0; n < OutputLayer.Neurons.size() - 1; ++n) {
            double Delta = TargetValues[n] - OutputLayer.Neurons[n].OutputValue;
            Error += Delta * Delta;
            
            OutputLayer.Neurons[n].Gradient = Delta * 
                ActivateDerivative(OutputLayer.Neurons[n].OutputValue, OutputLayer.Activation);
        }
        
        // RMS error
        Error = sqrt(Error / (OutputLayer.Neurons.size() - 1));
        
        // Calculate hidden layer gradients
        for (int LayerIdx = Layers.size() - 2; LayerIdx > 0; --LayerIdx) {
            auto& HiddenLayer = Layers[LayerIdx];
            auto& NextLayer = Layers[LayerIdx + 1];
            
            for (unsigned n = 0; n < HiddenLayer.Neurons.size() - 1; ++n) {
                double Sum = 0.0;
                
                // Sum contribution to errors of the next layer
                for (unsigned NextN = 0; NextN < NextLayer.Neurons.size() - 1; ++NextN) {
                    Sum += HiddenLayer.Neurons[n].OutputWeights[NextN].Weight * 
                           NextLayer.Neurons[NextN].Gradient;
                }
                
                HiddenLayer.Neurons[n].Gradient = Sum * 
                    ActivateDerivative(HiddenLayer.Neurons[n].OutputValue, HiddenLayer.Activation);
            }
        }
        
        // Update connection weights
        for (unsigned LayerIdx = 0; LayerIdx < Layers.size() - 1; ++LayerIdx) {
            auto& Layer = Layers[LayerIdx];
            auto& NextLayer = Layers[LayerIdx + 1];
            
            for (unsigned n = 0; n < Layer.Neurons.size(); ++n) {
                for (unsigned NextN = 0; NextN < NextLayer.Neurons.size() - 1; ++NextN) {
                    double DeltaWeight = 
                        LearningRate * NextLayer.Neurons[NextN].Gradient * Layer.Neurons[n].OutputValue +
                        Momentum * Layer.Neurons[n].OutputWeights[NextN].DeltaWeight;
                    
                    Layer.Neurons[n].OutputWeights[NextN].DeltaWeight = DeltaWeight;
                    Layer.Neurons[n].OutputWeights[NextN].Weight += DeltaWeight;
                }
            }
        }
    }
    
public:
    /**
     * @brief Construct a new Health Score Predictor
     * 
     * @param feature_list List of features used for prediction
     * @param hidden_layers Vector defining hidden layer sizes
     * @param hidden_activation Activation function for hidden layers
     * @param output_activation Activation function for output layer
     * @param learn_rate Learning rate for training
     * @param momentum_factor Momentum factor for training
     * 
     * Time Complexity: O(N*M) where N is number of layers and M is max neurons per layer
     * Space Complexity: O(N*M^2) for weights between layers
     */
    HealthScorePredictor(
        const std::vector<Feature>& FeatureList,
        const std::vector<unsigned>& HiddenLayers,
        ActivationType HiddenActivation = ActivationType::RELU,
        ActivationType OutputActivation = ActivationType::SIGMOID,
        double LearnRate = 0.01,
        double MomentumFactor = 0.9
    ) : Features(FeatureList), LearningRate(LearnRate), Momentum(MomentumFactor), Error(0.0), Epochs(0) {
        
        if (FeatureList.empty()) {
            throw std::invalid_argument("Feature list cannot be empty");
        }
        
        // Configure network topology
        unsigned NumFeatures = FeatureList.size();
        std::vector<unsigned> Topology;
        
        // Input layer (one neuron per feature)
        Topology.push_back(NumFeatures);
        
        // Hidden layers
        for (auto size : HiddenLayers) {
            Topology.push_back(size);
        }
        
        // Output layer (single neuron for health score)
        Topology.push_back(1);
        
        // Create network layers
        Layers.resize(Topology.size());
        
        for (unsigned i = 0; i < Layers.size(); ++i) {
            unsigned NumOutputs = (i == Topology.size() - 1) ? 0 : Topology[i + 1];
            
            // Set appropriate activation function
            ActivationType Activation;
            if (i == Layers.size() - 1) {
                Activation = OutputActivation;
            } else {
                Activation = HiddenActivation;
            }
            
            Layers[i] = Layer(Topology[i], NumOutputs, Activation);
        }
    }
    
    /**
     * @brief Train the network on a dataset
     * 
     * @param training_data Vector of feature vectors
     * @param target_values Vector of target health scores
     * @param batch_size Size of mini-batches
     * @param max_epochs Maximum number of training epochs
     * @param early_stopping_patience Number of epochs with no improvement before stopping
     * @param validation_split Fraction of data to use for validation
     * @return double Final validation error
     * 
     * Time Complexity: O(E*N*B) where E is epochs, N is samples, B is batch size
     * Space Complexity: O(N) for storing training data
     */
    double Train(
        const std::vector<std::vector<double>>& TrainingData,
        const std::vector<double>& TargetValues,
        unsigned BatchSize = 32,
        unsigned MaxEpochs = 1000,
        unsigned EarlyStoppingPatience = 20,
        double ValidationSplit = 0.2
    ) {
        if (TrainingData.empty() || TargetValues.empty() || TrainingData.size() != TargetValues.size()) {
            throw std::invalid_argument("Invalid training data or target values");
        }
        
        // Shuffle and split data for training and validation
        std::vector<size_t> Indices(TrainingData.size());
        std::iota(Indices.begin(), Indices.end(), 0);
        
        std::random_device Rd;
        std::mt19937 g(Rd());
        std::shuffle(Indices.begin(), Indices.end(), g);
        
        size_t ValidationSize = static_cast<size_t>(TrainingData.size() * ValidationSplit);
        size_t TrainingSize = TrainingData.size() - ValidationSize;
        
        // Prepare normalized data
        std::vector<std::vector<double>> NormTrainingData(TrainingData.size());
        for (size_t i = 0; i < TrainingData.size(); ++i) {
            NormTrainingData[i].resize(TrainingData[i].size());
            for (size_t j = 0; j < TrainingData[i].size(); ++j) {
                NormTrainingData[i][j] = Features[j].Normalize(TrainingData[i][j]);
            }
        }
        
        std::vector<double> NormTargets(TargetValues.size());
        for (size_t i = 0; i < TargetValues.size(); ++i) {
            // Assume health score is between 0 and 100, normalize to 0-1
            NormTargets[i] = TargetValues[i] / 100.0;
        }
        
        double BestValidationError = std::numeric_limits<double>::max();
        unsigned PatienceCounter = 0;
        
        for (Epochs = 0; Epochs < MaxEpochs; ++Epochs) {
            // Train on mini-batches
            for (size_t BatchStart = 0; BatchStart < TrainingSize; BatchStart += BatchSize) {
                size_t BatchEnd = std::min(BatchStart + BatchSize, TrainingSize);
                
                for (size_t i = BatchStart; i < BatchEnd; ++i) {
                    size_t Idx = Indices[i];
                    
                    // Forward propagation
                    FeedForward(NormTrainingData[Idx]);
                    
                    // Backpropagation
                    std::vector<double> Target = {NormTargets[Idx]};
                    BackPropagate(Target);
                }
            }
            
            // Validate
            double ValidationError = 0.0;
            for (size_t i = TrainingSize; i < TrainingData.size(); ++i) {
                size_t Idx = Indices[i];
                
                FeedForward(NormTrainingData[Idx]);
                
                // Calculate error
                double Output = Layers.back().Neurons[0].OutputValue;
                double Target = NormTargets[Idx];
                double Err = Target - Output;
                ValidationError += Err * Err;
            }
            
            ValidationError = sqrt(ValidationError / ValidationSize);
            
            // Early stopping check
            if (ValidationError < BestValidationError) {
                BestValidationError = ValidationError;
                PatienceCounter = 0;
            } else {
                PatienceCounter++;
                if (PatienceCounter >= EarlyStoppingPatience) {
                    break;
                }
            }
            
            // Report progress occasionally
            if (Epochs % 100 == 0) {
                std::cout << "Epoch " << Epochs << ": validation error = " << ValidationError << std::endl;
            }
        }
        
        std::cout << "Training completed after " << Epochs << " epochs. "
                  << "Final validation error: " << BestValidationError << std::endl;
        
        return BestValidationError;
    }
    
    /**
     * @brief Predict health score for a set of feature values
     * 
     * @param feature_values Vector of feature values in the same order as features
     * @return double Predicted health score (0-100)
     * 
     * This method is thread-safe and can be called from multiple threads.
     * 
     * Time Complexity: O(N) where N is the total number of neurons
     * Space Complexity: O(N) for temporary copy of layers
     */
    double Predict(const std::vector<double>& FeatureValues) const {
        if (FeatureValues.size() != Features.size()) {
            throw std::invalid_argument("Invalid number of feature values");
        }
        
        // Normalize input features
        std::vector<double> NormalizedValues(FeatureValues.size());
        for (size_t i = 0; i < FeatureValues.size(); ++i) {
            NormalizedValues[i] = Features[i].Normalize(FeatureValues[i]);
        }
        
        // Instead of copying the entire network (which won't work due to non-copyable mutex),
        // we'll create a thread-local copy of just the layers
        std::vector<Layer> LayersCopy = Layers;
        
        // Lock to ensure thread safety while computing the prediction
        // The mutex is declared mutable to allow locking in const methods
        std::lock_guard<std::mutex> Lock(InferenceMutex);
        
        // Forward propagation using our copied layers
        // First set input layer values
        for (unsigned i = 0; i < NormalizedValues.size(); ++i) {
            LayersCopy[0].Neurons[i].OutputValue = NormalizedValues[i];
        }
        
        // Then forward propagate through all subsequent layers
        for (unsigned LayerIdx = 1; LayerIdx < LayersCopy.size(); ++LayerIdx) {
            auto& PrevLayer = LayersCopy[LayerIdx - 1];
            auto& CurrentLayer = LayersCopy[LayerIdx];
            
            // Process all neurons except the bias neuron
            for (unsigned n = 0; n < CurrentLayer.Neurons.size() - 1; ++n) {
                double Sum = 0.0;
                
                // Sum weighted inputs from previous layer
                for (unsigned PrevN = 0; PrevN < PrevLayer.Neurons.size(); ++PrevN) {
                    Sum += PrevLayer.Neurons[PrevN].OutputValue * 
                           PrevLayer.Neurons[PrevN].OutputWeights[n].Weight;
                }
                
                // Apply activation function
                CurrentLayer.Neurons[n].OutputValue = 
                    Activate(Sum, CurrentLayer.Activation);
            }
        }
        
        // Get output and denormalize to health score (0-100)
        double NormalizedOutput = LayersCopy.back().Neurons[0].OutputValue;
        return NormalizedOutput * 100.0;
    }
    
    /**
     * @brief Save model to a file
     * 
     * @param filename File to save the model to
     * @return bool Success indicator
     */
    bool SaveModel(const std::string& Filename) const {
        std::ofstream File(Filename);
        if (!File.is_open()) {
            return false;
        }
        
        // Save feature metadata
        File << Features.size() << std::endl;
        for (const auto& Feature : Features) {
            File << Feature.Name << "," 
                 << Feature.MinValue << "," 
                 << Feature.MaxValue << "," 
                 << Feature.Mean << "," 
                 << Feature.StdDev << std::endl;
        }
        
        // Save network topology
        File << Layers.size() << std::endl;
        for (const auto& Layer : Layers) {
            File << Layer.Neurons.size() - 1 << "," 
                 << static_cast<int>(Layer.Activation) << std::endl;
        }
        
        // Save weights
        for (size_t i = 0; i < Layers.size() - 1; ++i) {
            for (const auto& Neuron : Layers[i].Neurons) {
                for (const auto& Connection : Neuron.OutputWeights) {
                    File << Connection.Weight << ",";
                }
                File << std::endl;
            }
        }
        
        // Save training metadata
        File << LearningRate << "," << Momentum << "," << Epochs << std::endl;
        
        return true;
    }
    
    /**
     * @brief Load model from a file
     * 
     * @param filename File to load the model from
     * @return bool Success indicator
     */
    bool LoadModel(const std::string& Filename) {
        std::ifstream File(Filename);
        if (!File.is_open()) {
            return false;
        }
        
        std::string Line;
        
        // Load feature metadata
        std::getline(File, Line);
        size_t NumFeatures = std::stoul(Line);
        Features.resize(NumFeatures);
        
        for (size_t i = 0; i < NumFeatures; ++i) {
            std::getline(File, Line);
            std::stringstream Ss(Line);
            std::string Name, MinVal, MaxVal, MeanVal, StdVal;
            
            std::getline(Ss, Name, ',');
            std::getline(Ss, MinVal, ',');
            std::getline(Ss, MaxVal, ',');
            std::getline(Ss, MeanVal, ',');
            std::getline(Ss, StdVal, ',');
            
            Features[i].Name = Name;
            Features[i].MinValue = std::stod(MinVal);
            Features[i].MaxValue = std::stod(MaxVal);
            Features[i].Mean = std::stod(MeanVal);
            Features[i].StdDev = std::stod(StdVal);
        }
        
        // Load network topology
        std::getline(File, Line);
        size_t NumLayers = std::stoul(Line);
        
        std::vector<unsigned> Topology;
        std::vector<ActivationType> Activations;
        
        for (size_t i = 0; i < NumLayers; ++i) {
            std::getline(File, Line);
            std::stringstream Ss(Line);
            std::string SizeStr, ActStr;
            
            std::getline(Ss, SizeStr, ',');
            std::getline(Ss, ActStr, ',');
            
            Topology.push_back(std::stoul(SizeStr));
            Activations.push_back(static_cast<ActivationType>(std::stoi(ActStr)));
        }
        
        // Recreate network with loaded topology
        Layers.resize(Topology.size());
        for (unsigned i = 0; i < Layers.size(); ++i) {
            unsigned NumOutputs = (i == Topology.size() - 1) ? 0 : Topology[i + 1];
            Layers[i] = Layer(Topology[i], NumOutputs, Activations[i]);
        }
        
        // Load weights
        for (size_t i = 0; i < Layers.size() - 1; ++i) {
            for (auto& Neuron : Layers[i].Neurons) {
                std::getline(File, Line);
                std::stringstream Ss(Line);
                std::string WeightStr;
                
                for (size_t j = 0; j < Neuron.OutputWeights.size(); ++j) {
                    std::getline(Ss, WeightStr, ',');
                    Neuron.OutputWeights[j].Weight = std::stod(WeightStr);
                }
            }
        }
        
        // Load training metadata
        std::getline(File, Line);
        std::stringstream Ss(Line);
        std::string LrStr, MomentumStr, EpochsStr;
        
        std::getline(Ss, LrStr, ',');
        std::getline(Ss, MomentumStr, ',');
        std::getline(Ss, EpochsStr, ',');
        
        LearningRate = std::stod(LrStr);
        Momentum = std::stod(MomentumStr);
        Epochs = std::stoul(EpochsStr);
        
        return true;
    }
    
    /**
     * @brief Get feature importance by analyzing weights
     * 
     * @return std::vector<std::pair<std::string, double>> Feature names and importance scores
     */
    std::vector<std::pair<std::string, double>> GetFeatureImportance() const {
        std::vector<std::pair<std::string, double>> Importance;
        
        // Calculate importance based on the sum of absolute weights
        // connecting each input feature to the first hidden layer
        for (size_t i = 0; i < Features.size(); ++i) {
            double SumWeights = 0.0;
            for (const auto& Connection : Layers[0].Neurons[i].OutputWeights) {
                SumWeights += std::abs(Connection.Weight);
            }
            
            Importance.emplace_back(Features[i].Name, SumWeights);
        }
        
        // Sort by importance (descending)
        std::sort(Importance.begin(), Importance.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return Importance;
    }
    
    /**
     * @brief Explain prediction by showing feature contributions
     * 
     * @param feature_values Input feature values
     * @return std::vector<std::pair<std::string, double>> Feature contributions to the prediction
     */
    std::vector<std::pair<std::string, double>> ExplainPrediction(
        const std::vector<double>& FeatureValues) const {
        
        if (FeatureValues.size() != Features.size()) {
            throw std::invalid_argument("Invalid number of feature values");
        }
        
        std::vector<std::pair<std::string, double>> Contributions;
        
        // Calculate baseline prediction with all features at their means
        std::vector<double> BaselineValues(Features.size());
        for (size_t i = 0; i < Features.size(); ++i) {
            BaselineValues[i] = Features[i].Mean;
        }
        
        double BaselinePrediction = Predict(BaselineValues);
        
        // Calculate contribution of each feature
        for (size_t i = 0; i < Features.size(); ++i) {
            std::vector<double> TempValues = BaselineValues;
            TempValues[i] = FeatureValues[i];
            
            double NewPrediction = Predict(TempValues);
            double Contribution = NewPrediction - BaselinePrediction;
            
            Contributions.emplace_back(Features[i].Name, Contribution);
        }
        
        // Sort by absolute contribution (descending)
        std::sort(Contributions.begin(), Contributions.end(),
                 [](const auto& a, const auto& b) { 
                     return std::abs(a.second) > std::abs(b.second); 
                 });
        
        return Contributions;
    }
};

/**
 * @brief Helper class to load and preprocess CSV data for the health predictor
 * 
 * Provides utilities to:
 * - Load CSV files
 * - Calculate statistics for feature normalization
 * - Split data into training and test sets
 */
class HealthDataLoader {
public:
    /**
     * @brief Load health data from a CSV file
     * 
     * @param filename CSV file path
     * @param has_header Whether the CSV has a header row
     * @param target_column Name of the target column (health score)
     * @return std::pair<std::vector<HealthScorePredictor::Feature>, std::vector<std::vector<double>>> 
     *         Features and data matrix
     * 
     * Time Complexity: O(N*M) where N is rows and M is columns
     * Space Complexity: O(N*M) for storing the data
     */
    static std::pair<std::vector<HealthScorePredictor::Feature>, 
                    std::pair<std::vector<std::vector<double>>, std::vector<double>>> 
    LoadCsv(
        const std::string& Filename,
        bool HasHeader = true,
        const std::string& TargetColumn = "health_score"
    ) {
        std::ifstream File(Filename);
        if (!File.is_open()) {
            throw std::runtime_error("Failed to open file: " + Filename);
        }
        
        std::vector<std::string> Header;
        std::vector<std::vector<double>> data;
        std::vector<double> TargetValues;
        
        std::string Line;
        
        // Read header
        if (HasHeader && std::getline(File, Line)) {
            std::stringstream Ss(Line);
            std::string Cell;
            
            while (std::getline(Ss, Cell, ',')) {
                Header.push_back(Cell);
            }
        } else {
            // If no header, rewind file
            File.seekg(0, std::ios::beg);
        }
        
        // Find target column index
        int TargetIdx = -1;
        if (HasHeader) {
            for (size_t i = 0; i < Header.size(); ++i) {
                if (Header[i] == TargetColumn) {
                    TargetIdx = static_cast<int>(i);
                    break;
                }
            }
            
            if (TargetIdx < 0) {
                throw std::runtime_error("Target column not found: " + TargetColumn);
            }
        } else {
            // Assume last column is target if no header
            TargetIdx = -2; // Will be set after reading first row
        }
        
        // Read data
        while (std::getline(File, Line)) {
            std::stringstream Ss(Line);
            std::string Cell;
            std::vector<double> Row;
            int ColIdx = 0;
            
            while (std::getline(Ss, Cell, ',')) {
                double value = std::stod(Cell);
                
                if (TargetIdx == -2) {
                    // Set target index after reading first row
                    TargetIdx = ColIdx;
                }
                
                if (ColIdx == TargetIdx) {
                    TargetValues.push_back(value);
                } else {
                    Row.push_back(value);
                }
                
                ++ColIdx;
            }
            
            data.push_back(Row);
        }
        
        // Create feature definitions with statistics
        std::vector<HealthScorePredictor::Feature> Features;
        
        // Calculate statistics for each feature
        for (size_t Col = 0; Col < data[0].size(); ++Col) {
            HealthScorePredictor::Feature Feature;
            
            // Set feature name
            if (HasHeader) {
                // Skip target column in header indexing
                int HeaderIdx = (Col >= static_cast<size_t>(TargetIdx)) ? Col + 1 : Col;
                Feature.Name = Header[HeaderIdx];
            } else {
                Feature.Name = "Feature_" + std::to_string(Col);
            }
            
            // Calculate min, max, mean
            Feature.MinValue = std::numeric_limits<double>::max();
            Feature.MaxValue = std::numeric_limits<double>::lowest();
            double Sum = 0.0;
            
            for (const auto& Row : data) {
                Feature.MinValue = std::min(Feature.MinValue, Row[Col]);
                Feature.MaxValue = std::max(Feature.MaxValue, Row[Col]);
                Sum += Row[Col];
            }
            
            Feature.Mean = Sum / data.size();
            
            // Calculate standard deviation
            double VarianceSum = 0.0;
            for (const auto& Row : data) {
                double Diff = Row[Col] - Feature.Mean;
                VarianceSum += Diff * Diff;
            }
            
            Feature.StdDev = sqrt(VarianceSum / data.size());
            
            Features.push_back(Feature);
        }
        
        return {Features, {data, TargetValues}};
    }
    
    /**
     * @brief Split data into training and test sets
     * 
     * @param data Input data matrix
     * @param targets Target values
     * @param test_ratio Ratio of data to use for testing (0.0-1.0)
     * @return std::pair<std::pair<std::vector<std::vector<double>>, std::vector<double>>, 
     *                  std::pair<std::vector<std::vector<double>>, std::vector<double>>> 
     *         Training and test data with their targets
     */
    static std::pair<std::pair<std::vector<std::vector<double>>, std::vector<double>>,
                    std::pair<std::vector<std::vector<double>>, std::vector<double>>>
    TrainTestSplit(
        const std::vector<std::vector<double>>& data,
        const std::vector<double>& Targets,
        double TestRatio = 0.2
    ) {
        if (data.empty() || Targets.empty() || data.size() != Targets.size()) {
            throw std::invalid_argument("Invalid data or targets");
        }
        
        // Create indices and shuffle them
        std::vector<size_t> Indices(data.size());
        std::iota(Indices.begin(), Indices.end(), 0);
        
        std::random_device Rd;
        std::mt19937 g(Rd());
        std::shuffle(Indices.begin(), Indices.end(), g);
        
        // Calculate split point
        size_t TestSize = static_cast<size_t>(data.size() * TestRatio);
        size_t TrainSize = data.size() - TestSize;
        
        // Create training and test sets
        std::vector<std::vector<double>> TrainData, TestData;
        std::vector<double> TrainTargets, TestTargets;
        
        TrainData.reserve(TrainSize);
        TrainTargets.reserve(TrainSize);
        TestData.reserve(TestSize);
        TestTargets.reserve(TestSize);
        
        for (size_t i = 0; i < TrainSize; ++i) {
            TrainData.push_back(data[Indices[i]]);
            TrainTargets.push_back(Targets[Indices[i]]);
        }
        
        for (size_t i = TrainSize; i < data.size(); ++i) {
            TestData.push_back(data[Indices[i]]);
            TestTargets.push_back(Targets[Indices[i]]);
        }
        
        return {{TrainData, TrainTargets}, {TestData, TestTargets}};
    }
    
    /**
     * @brief Generate statistics for model evaluation
     * 
     * @param predictor Trained predictor model
     * @param test_data Test data matrix
     * @param test_targets Test target values
     * @return std::unordered_map<std::string, double> Statistics (MAE, MSE, RMSE, R²)
     */
    static std::unordered_map<std::string, double> EvaluateModel(
        const HealthScorePredictor& Predictor,
        const std::vector<std::vector<double>>& TestData,
        const std::vector<double>& TestTargets
    ) {
        if (TestData.empty() || TestTargets.empty() || TestData.size() != TestTargets.size()) {
            throw std::invalid_argument("Invalid test data or targets");
        }
        
        double Mae = 0.0;  // Mean Absolute Error
        double Mse = 0.0;  // Mean Squared Error
        double SumActual = 0.0;
        double SumSquaredActual = 0.0;
        std::vector<double> Predictions(TestTargets.size());
        
        // Calculate errors
        for (size_t i = 0; i < TestData.size(); ++i) {
            double Prediction = Predictor.Predict(TestData[i]);
            Predictions[i] = Prediction;
            
            double Error = Prediction - TestTargets[i];
            Mae += std::abs(Error);
            Mse += Error * Error;
            
            SumActual += TestTargets[i];
            SumSquaredActual += TestTargets[i] * TestTargets[i];
        }
        
        Mae /= TestData.size();
        Mse /= TestData.size();
        double Rmse = std::sqrt(Mse);
        
        // Calculate R² (coefficient of determination)
        double MeanActual = SumActual / TestTargets.size();
        double TotalVariance = 0.0;
        double ResidualVariance = 0.0;
        
        for (size_t i = 0; i < TestTargets.size(); ++i) {
            TotalVariance += std::pow(TestTargets[i] - MeanActual, 2);
            ResidualVariance += std::pow(TestTargets[i] - Predictions[i], 2);
        }
        
        double RSquared = 1.0 - (ResidualVariance / TotalVariance);
        
        return {
            {"MAE", Mae},
            {"MSE", Mse},
            {"RMSE", Rmse},
            {"R²", RSquared}
        };
    }
};

/**
 * @brief Main function to demonstrate the health score predictor
 * 
 * This example:
 * 1. Loads data from a CSV file
 * 2. Trains a neural network model
 * 3. Evaluates model performance
 * 4. Makes predictions for new individuals
 * 5. Explains feature importance
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @return int Exit code
 */
