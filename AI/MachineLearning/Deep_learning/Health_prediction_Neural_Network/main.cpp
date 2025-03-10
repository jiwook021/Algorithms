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
        std::string name;
        double min_value;
        double max_value;
        double mean;
        double std_dev;
        
        // Normalize a value using min-max scaling
        double normalize(double value) const {
            if (std_dev > 0) {
                // Z-score normalization
                return (value - mean) / std_dev;
            } else {
                // Min-max normalization as fallback
                return (value - min_value) / (max_value - min_value) * 2.0 - 1.0;
            }
        }
        
        // Denormalize a value
        double denormalize(double normalized_value) const {
            if (std_dev > 0) {
                // Z-score denormalization
                return normalized_value * std_dev + mean;
            } else {
                // Min-max denormalization
                return (normalized_value + 1.0) / 2.0 * (max_value - min_value) + min_value;
            }
        }
    };
    
private:
    // Neural network architecture
    struct Neuron;
    struct Layer;
    
    struct Connection {
        double weight;
        double delta_weight;
        
        Connection() : weight(0.0), delta_weight(0.0) {}
    };
    
    struct Neuron {
        std::vector<Connection> output_weights;
        double output_value;
        double gradient;
        unsigned index;
        
        Neuron(unsigned num_outputs, unsigned idx) : output_value(0.0), gradient(0.0), index(idx) {
            // Initialize with random weights
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::normal_distribution<double> dist(0.0, 1.0);
            
            output_weights.resize(num_outputs);
            for (auto& connection : output_weights) {
                // Xavier/Glorot initialization
                connection.weight = dist(gen) * sqrt(2.0 / (num_outputs + 1));
            }
        }
    };
    
    struct Layer {
        std::vector<Neuron> neurons;
        ActivationType activation;
        
        Layer() = default;
        
        Layer(unsigned num_neurons, unsigned num_outputs, ActivationType act)
            : activation(act) {
            neurons.reserve(num_neurons + 1); // +1 for bias neuron
            
            // Create neurons in the layer
            for (unsigned i = 0; i < num_neurons; ++i) {
                neurons.emplace_back(num_outputs, i);
            }
            
            // Add bias neuron (with constant output of 1.0)
            neurons.emplace_back(num_outputs, num_neurons);
            neurons.back().output_value = 1.0;
        }
    };
    
    std::vector<Layer> layers;
    std::vector<Feature> features;
    double learning_rate;
    double momentum;
    double error;
    unsigned epochs;
    mutable std::mutex inference_mutex; // For thread-safe inference, mutable to allow const methods to lock it
    
    // Activation functions
    double activate(double x, ActivationType type) const {
        switch (type) {
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
    double activate_derivative(double x, ActivationType type) const {
        switch (type) {
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
    void feed_forward(const std::vector<double>& input_values) {
        // Set input layer values
        for (unsigned i = 0; i < input_values.size(); ++i) {
            layers[0].neurons[i].output_value = input_values[i];
        }
        
        // Forward propagate through all subsequent layers
        for (unsigned layer_idx = 1; layer_idx < layers.size(); ++layer_idx) {
            auto& prev_layer = layers[layer_idx - 1];
            auto& current_layer = layers[layer_idx];
            
            // Process all neurons except the bias neuron
            for (unsigned n = 0; n < current_layer.neurons.size() - 1; ++n) {
                double sum = 0.0;
                
                // Sum weighted inputs from previous layer
                for (unsigned prev_n = 0; prev_n < prev_layer.neurons.size(); ++prev_n) {
                    sum += prev_layer.neurons[prev_n].output_value * 
                           prev_layer.neurons[prev_n].output_weights[n].weight;
                }
                
                // Apply activation function
                current_layer.neurons[n].output_value = 
                    activate(sum, current_layer.activation);
            }
        }
    }
    
    // Backpropagation
    void back_propagate(const std::vector<double>& target_values) {
        // Calculate output layer gradients
        auto& output_layer = layers.back();
        error = 0.0;
        
        for (unsigned n = 0; n < output_layer.neurons.size() - 1; ++n) {
            double delta = target_values[n] - output_layer.neurons[n].output_value;
            error += delta * delta;
            
            output_layer.neurons[n].gradient = delta * 
                activate_derivative(output_layer.neurons[n].output_value, output_layer.activation);
        }
        
        // RMS error
        error = sqrt(error / (output_layer.neurons.size() - 1));
        
        // Calculate hidden layer gradients
        for (int layer_idx = layers.size() - 2; layer_idx > 0; --layer_idx) {
            auto& hidden_layer = layers[layer_idx];
            auto& next_layer = layers[layer_idx + 1];
            
            for (unsigned n = 0; n < hidden_layer.neurons.size() - 1; ++n) {
                double sum = 0.0;
                
                // Sum contribution to errors of the next layer
                for (unsigned next_n = 0; next_n < next_layer.neurons.size() - 1; ++next_n) {
                    sum += hidden_layer.neurons[n].output_weights[next_n].weight * 
                           next_layer.neurons[next_n].gradient;
                }
                
                hidden_layer.neurons[n].gradient = sum * 
                    activate_derivative(hidden_layer.neurons[n].output_value, hidden_layer.activation);
            }
        }
        
        // Update connection weights
        for (unsigned layer_idx = 0; layer_idx < layers.size() - 1; ++layer_idx) {
            auto& layer = layers[layer_idx];
            auto& next_layer = layers[layer_idx + 1];
            
            for (unsigned n = 0; n < layer.neurons.size(); ++n) {
                for (unsigned next_n = 0; next_n < next_layer.neurons.size() - 1; ++next_n) {
                    double delta_weight = 
                        learning_rate * next_layer.neurons[next_n].gradient * layer.neurons[n].output_value +
                        momentum * layer.neurons[n].output_weights[next_n].delta_weight;
                    
                    layer.neurons[n].output_weights[next_n].delta_weight = delta_weight;
                    layer.neurons[n].output_weights[next_n].weight += delta_weight;
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
        const std::vector<Feature>& feature_list,
        const std::vector<unsigned>& hidden_layers,
        ActivationType hidden_activation = ActivationType::RELU,
        ActivationType output_activation = ActivationType::SIGMOID,
        double learn_rate = 0.01,
        double momentum_factor = 0.9
    ) : features(feature_list), learning_rate(learn_rate), momentum(momentum_factor), error(0.0), epochs(0) {
        
        if (feature_list.empty()) {
            throw std::invalid_argument("Feature list cannot be empty");
        }
        
        // Configure network topology
        unsigned num_features = feature_list.size();
        std::vector<unsigned> topology;
        
        // Input layer (one neuron per feature)
        topology.push_back(num_features);
        
        // Hidden layers
        for (auto size : hidden_layers) {
            topology.push_back(size);
        }
        
        // Output layer (single neuron for health score)
        topology.push_back(1);
        
        // Create network layers
        layers.resize(topology.size());
        
        for (unsigned i = 0; i < layers.size(); ++i) {
            unsigned num_outputs = (i == topology.size() - 1) ? 0 : topology[i + 1];
            
            // Set appropriate activation function
            ActivationType activation;
            if (i == layers.size() - 1) {
                activation = output_activation;
            } else {
                activation = hidden_activation;
            }
            
            layers[i] = Layer(topology[i], num_outputs, activation);
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
    double train(
        const std::vector<std::vector<double>>& training_data,
        const std::vector<double>& target_values,
        unsigned batch_size = 32,
        unsigned max_epochs = 1000,
        unsigned early_stopping_patience = 20,
        double validation_split = 0.2
    ) {
        if (training_data.empty() || target_values.empty() || training_data.size() != target_values.size()) {
            throw std::invalid_argument("Invalid training data or target values");
        }
        
        // Shuffle and split data for training and validation
        std::vector<size_t> indices(training_data.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        size_t validation_size = static_cast<size_t>(training_data.size() * validation_split);
        size_t training_size = training_data.size() - validation_size;
        
        // Prepare normalized data
        std::vector<std::vector<double>> norm_training_data(training_data.size());
        for (size_t i = 0; i < training_data.size(); ++i) {
            norm_training_data[i].resize(training_data[i].size());
            for (size_t j = 0; j < training_data[i].size(); ++j) {
                norm_training_data[i][j] = features[j].normalize(training_data[i][j]);
            }
        }
        
        std::vector<double> norm_targets(target_values.size());
        for (size_t i = 0; i < target_values.size(); ++i) {
            // Assume health score is between 0 and 100, normalize to 0-1
            norm_targets[i] = target_values[i] / 100.0;
        }
        
        double best_validation_error = std::numeric_limits<double>::max();
        unsigned patience_counter = 0;
        
        for (epochs = 0; epochs < max_epochs; ++epochs) {
            // Train on mini-batches
            for (size_t batch_start = 0; batch_start < training_size; batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, training_size);
                
                for (size_t i = batch_start; i < batch_end; ++i) {
                    size_t idx = indices[i];
                    
                    // Forward propagation
                    feed_forward(norm_training_data[idx]);
                    
                    // Backpropagation
                    std::vector<double> target = {norm_targets[idx]};
                    back_propagate(target);
                }
            }
            
            // Validate
            double validation_error = 0.0;
            for (size_t i = training_size; i < training_data.size(); ++i) {
                size_t idx = indices[i];
                
                feed_forward(norm_training_data[idx]);
                
                // Calculate error
                double output = layers.back().neurons[0].output_value;
                double target = norm_targets[idx];
                double err = target - output;
                validation_error += err * err;
            }
            
            validation_error = sqrt(validation_error / validation_size);
            
            // Early stopping check
            if (validation_error < best_validation_error) {
                best_validation_error = validation_error;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= early_stopping_patience) {
                    break;
                }
            }
            
            // Report progress occasionally
            if (epochs % 100 == 0) {
                std::cout << "Epoch " << epochs << ": validation error = " << validation_error << std::endl;
            }
        }
        
        std::cout << "Training completed after " << epochs << " epochs. "
                  << "Final validation error: " << best_validation_error << std::endl;
        
        return best_validation_error;
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
    double predict(const std::vector<double>& feature_values) const {
        if (feature_values.size() != features.size()) {
            throw std::invalid_argument("Invalid number of feature values");
        }
        
        // Normalize input features
        std::vector<double> normalized_values(feature_values.size());
        for (size_t i = 0; i < feature_values.size(); ++i) {
            normalized_values[i] = features[i].normalize(feature_values[i]);
        }
        
        // Instead of copying the entire network (which won't work due to non-copyable mutex),
        // we'll create a thread-local copy of just the layers
        std::vector<Layer> layers_copy = layers;
        
        // Lock to ensure thread safety while computing the prediction
        // The mutex is declared mutable to allow locking in const methods
        std::lock_guard<std::mutex> lock(inference_mutex);
        
        // Forward propagation using our copied layers
        // First set input layer values
        for (unsigned i = 0; i < normalized_values.size(); ++i) {
            layers_copy[0].neurons[i].output_value = normalized_values[i];
        }
        
        // Then forward propagate through all subsequent layers
        for (unsigned layer_idx = 1; layer_idx < layers_copy.size(); ++layer_idx) {
            auto& prev_layer = layers_copy[layer_idx - 1];
            auto& current_layer = layers_copy[layer_idx];
            
            // Process all neurons except the bias neuron
            for (unsigned n = 0; n < current_layer.neurons.size() - 1; ++n) {
                double sum = 0.0;
                
                // Sum weighted inputs from previous layer
                for (unsigned prev_n = 0; prev_n < prev_layer.neurons.size(); ++prev_n) {
                    sum += prev_layer.neurons[prev_n].output_value * 
                           prev_layer.neurons[prev_n].output_weights[n].weight;
                }
                
                // Apply activation function
                current_layer.neurons[n].output_value = 
                    activate(sum, current_layer.activation);
            }
        }
        
        // Get output and denormalize to health score (0-100)
        double normalized_output = layers_copy.back().neurons[0].output_value;
        return normalized_output * 100.0;
    }
    
    /**
     * @brief Save model to a file
     * 
     * @param filename File to save the model to
     * @return bool Success indicator
     */
    bool save_model(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        // Save feature metadata
        file << features.size() << std::endl;
        for (const auto& feature : features) {
            file << feature.name << "," 
                 << feature.min_value << "," 
                 << feature.max_value << "," 
                 << feature.mean << "," 
                 << feature.std_dev << std::endl;
        }
        
        // Save network topology
        file << layers.size() << std::endl;
        for (const auto& layer : layers) {
            file << layer.neurons.size() - 1 << "," 
                 << static_cast<int>(layer.activation) << std::endl;
        }
        
        // Save weights
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            for (const auto& neuron : layers[i].neurons) {
                for (const auto& connection : neuron.output_weights) {
                    file << connection.weight << ",";
                }
                file << std::endl;
            }
        }
        
        // Save training metadata
        file << learning_rate << "," << momentum << "," << epochs << std::endl;
        
        return true;
    }
    
    /**
     * @brief Load model from a file
     * 
     * @param filename File to load the model from
     * @return bool Success indicator
     */
    bool load_model(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::string line;
        
        // Load feature metadata
        std::getline(file, line);
        size_t num_features = std::stoul(line);
        features.resize(num_features);
        
        for (size_t i = 0; i < num_features; ++i) {
            std::getline(file, line);
            std::stringstream ss(line);
            std::string name, min_val, max_val, mean_val, std_val;
            
            std::getline(ss, name, ',');
            std::getline(ss, min_val, ',');
            std::getline(ss, max_val, ',');
            std::getline(ss, mean_val, ',');
            std::getline(ss, std_val, ',');
            
            features[i].name = name;
            features[i].min_value = std::stod(min_val);
            features[i].max_value = std::stod(max_val);
            features[i].mean = std::stod(mean_val);
            features[i].std_dev = std::stod(std_val);
        }
        
        // Load network topology
        std::getline(file, line);
        size_t num_layers = std::stoul(line);
        
        std::vector<unsigned> topology;
        std::vector<ActivationType> activations;
        
        for (size_t i = 0; i < num_layers; ++i) {
            std::getline(file, line);
            std::stringstream ss(line);
            std::string size_str, act_str;
            
            std::getline(ss, size_str, ',');
            std::getline(ss, act_str, ',');
            
            topology.push_back(std::stoul(size_str));
            activations.push_back(static_cast<ActivationType>(std::stoi(act_str)));
        }
        
        // Recreate network with loaded topology
        layers.resize(topology.size());
        for (unsigned i = 0; i < layers.size(); ++i) {
            unsigned num_outputs = (i == topology.size() - 1) ? 0 : topology[i + 1];
            layers[i] = Layer(topology[i], num_outputs, activations[i]);
        }
        
        // Load weights
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            for (auto& neuron : layers[i].neurons) {
                std::getline(file, line);
                std::stringstream ss(line);
                std::string weight_str;
                
                for (size_t j = 0; j < neuron.output_weights.size(); ++j) {
                    std::getline(ss, weight_str, ',');
                    neuron.output_weights[j].weight = std::stod(weight_str);
                }
            }
        }
        
        // Load training metadata
        std::getline(file, line);
        std::stringstream ss(line);
        std::string lr_str, momentum_str, epochs_str;
        
        std::getline(ss, lr_str, ',');
        std::getline(ss, momentum_str, ',');
        std::getline(ss, epochs_str, ',');
        
        learning_rate = std::stod(lr_str);
        momentum = std::stod(momentum_str);
        epochs = std::stoul(epochs_str);
        
        return true;
    }
    
    /**
     * @brief Get feature importance by analyzing weights
     * 
     * @return std::vector<std::pair<std::string, double>> Feature names and importance scores
     */
    std::vector<std::pair<std::string, double>> get_feature_importance() const {
        std::vector<std::pair<std::string, double>> importance;
        
        // Calculate importance based on the sum of absolute weights
        // connecting each input feature to the first hidden layer
        for (size_t i = 0; i < features.size(); ++i) {
            double sum_weights = 0.0;
            for (const auto& connection : layers[0].neurons[i].output_weights) {
                sum_weights += std::abs(connection.weight);
            }
            
            importance.emplace_back(features[i].name, sum_weights);
        }
        
        // Sort by importance (descending)
        std::sort(importance.begin(), importance.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return importance;
    }
    
    /**
     * @brief Explain prediction by showing feature contributions
     * 
     * @param feature_values Input feature values
     * @return std::vector<std::pair<std::string, double>> Feature contributions to the prediction
     */
    std::vector<std::pair<std::string, double>> explain_prediction(
        const std::vector<double>& feature_values) const {
        
        if (feature_values.size() != features.size()) {
            throw std::invalid_argument("Invalid number of feature values");
        }
        
        std::vector<std::pair<std::string, double>> contributions;
        
        // Calculate baseline prediction with all features at their means
        std::vector<double> baseline_values(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            baseline_values[i] = features[i].mean;
        }
        
        double baseline_prediction = predict(baseline_values);
        
        // Calculate contribution of each feature
        for (size_t i = 0; i < features.size(); ++i) {
            std::vector<double> temp_values = baseline_values;
            temp_values[i] = feature_values[i];
            
            double new_prediction = predict(temp_values);
            double contribution = new_prediction - baseline_prediction;
            
            contributions.emplace_back(features[i].name, contribution);
        }
        
        // Sort by absolute contribution (descending)
        std::sort(contributions.begin(), contributions.end(),
                 [](const auto& a, const auto& b) { 
                     return std::abs(a.second) > std::abs(b.second); 
                 });
        
        return contributions;
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
    load_csv(
        const std::string& filename,
        bool has_header = true,
        const std::string& target_column = "health_score"
    ) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        
        std::vector<std::string> header;
        std::vector<std::vector<double>> data;
        std::vector<double> target_values;
        
        std::string line;
        
        // Read header
        if (has_header && std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, ',')) {
                header.push_back(cell);
            }
        } else {
            // If no header, rewind file
            file.seekg(0, std::ios::beg);
        }
        
        // Find target column index
        int target_idx = -1;
        if (has_header) {
            for (size_t i = 0; i < header.size(); ++i) {
                if (header[i] == target_column) {
                    target_idx = static_cast<int>(i);
                    break;
                }
            }
            
            if (target_idx < 0) {
                throw std::runtime_error("Target column not found: " + target_column);
            }
        } else {
            // Assume last column is target if no header
            target_idx = -2; // Will be set after reading first row
        }
        
        // Read data
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;
            int col_idx = 0;
            
            while (std::getline(ss, cell, ',')) {
                double value = std::stod(cell);
                
                if (target_idx == -2) {
                    // Set target index after reading first row
                    target_idx = col_idx;
                }
                
                if (col_idx == target_idx) {
                    target_values.push_back(value);
                } else {
                    row.push_back(value);
                }
                
                ++col_idx;
            }
            
            data.push_back(row);
        }
        
        // Create feature definitions with statistics
        std::vector<HealthScorePredictor::Feature> features;
        
        // Calculate statistics for each feature
        for (size_t col = 0; col < data[0].size(); ++col) {
            HealthScorePredictor::Feature feature;
            
            // Set feature name
            if (has_header) {
                // Skip target column in header indexing
                int header_idx = (col >= static_cast<size_t>(target_idx)) ? col + 1 : col;
                feature.name = header[header_idx];
            } else {
                feature.name = "Feature_" + std::to_string(col);
            }
            
            // Calculate min, max, mean
            feature.min_value = std::numeric_limits<double>::max();
            feature.max_value = std::numeric_limits<double>::lowest();
            double sum = 0.0;
            
            for (const auto& row : data) {
                feature.min_value = std::min(feature.min_value, row[col]);
                feature.max_value = std::max(feature.max_value, row[col]);
                sum += row[col];
            }
            
            feature.mean = sum / data.size();
            
            // Calculate standard deviation
            double variance_sum = 0.0;
            for (const auto& row : data) {
                double diff = row[col] - feature.mean;
                variance_sum += diff * diff;
            }
            
            feature.std_dev = sqrt(variance_sum / data.size());
            
            features.push_back(feature);
        }
        
        return {features, {data, target_values}};
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
    train_test_split(
        const std::vector<std::vector<double>>& data,
        const std::vector<double>& targets,
        double test_ratio = 0.2
    ) {
        if (data.empty() || targets.empty() || data.size() != targets.size()) {
            throw std::invalid_argument("Invalid data or targets");
        }
        
        // Create indices and shuffle them
        std::vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Calculate split point
        size_t test_size = static_cast<size_t>(data.size() * test_ratio);
        size_t train_size = data.size() - test_size;
        
        // Create training and test sets
        std::vector<std::vector<double>> train_data, test_data;
        std::vector<double> train_targets, test_targets;
        
        train_data.reserve(train_size);
        train_targets.reserve(train_size);
        test_data.reserve(test_size);
        test_targets.reserve(test_size);
        
        for (size_t i = 0; i < train_size; ++i) {
            train_data.push_back(data[indices[i]]);
            train_targets.push_back(targets[indices[i]]);
        }
        
        for (size_t i = train_size; i < data.size(); ++i) {
            test_data.push_back(data[indices[i]]);
            test_targets.push_back(targets[indices[i]]);
        }
        
        return {{train_data, train_targets}, {test_data, test_targets}};
    }
    
    /**
     * @brief Generate statistics for model evaluation
     * 
     * @param predictor Trained predictor model
     * @param test_data Test data matrix
     * @param test_targets Test target values
     * @return std::unordered_map<std::string, double> Statistics (MAE, MSE, RMSE, R²)
     */
    static std::unordered_map<std::string, double> evaluate_model(
        const HealthScorePredictor& predictor,
        const std::vector<std::vector<double>>& test_data,
        const std::vector<double>& test_targets
    ) {
        if (test_data.empty() || test_targets.empty() || test_data.size() != test_targets.size()) {
            throw std::invalid_argument("Invalid test data or targets");
        }
        
        double mae = 0.0;  // Mean Absolute Error
        double mse = 0.0;  // Mean Squared Error
        double sum_actual = 0.0;
        double sum_squared_actual = 0.0;
        std::vector<double> predictions(test_targets.size());
        
        // Calculate errors
        for (size_t i = 0; i < test_data.size(); ++i) {
            double prediction = predictor.predict(test_data[i]);
            predictions[i] = prediction;
            
            double error = prediction - test_targets[i];
            mae += std::abs(error);
            mse += error * error;
            
            sum_actual += test_targets[i];
            sum_squared_actual += test_targets[i] * test_targets[i];
        }
        
        mae /= test_data.size();
        mse /= test_data.size();
        double rmse = std::sqrt(mse);
        
        // Calculate R² (coefficient of determination)
        double mean_actual = sum_actual / test_targets.size();
        double total_variance = 0.0;
        double residual_variance = 0.0;
        
        for (size_t i = 0; i < test_targets.size(); ++i) {
            total_variance += std::pow(test_targets[i] - mean_actual, 2);
            residual_variance += std::pow(test_targets[i] - predictions[i], 2);
        }
        
        double r_squared = 1.0 - (residual_variance / total_variance);
        
        return {
            {"MAE", mae},
            {"MSE", mse},
            {"RMSE", rmse},
            {"R²", r_squared}
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
int main(int argc, char* argv[]) {
    try {
        std::string filename = (argc > 1) ? argv[1] : "health_data.csv";
        
        std::cout << "Loading data from " << filename << "..." << std::endl;
        
        // Load data
        auto [features, data_pair] = HealthDataLoader::load_csv(filename);
        auto& [data, targets] = data_pair;
        
        std::cout << "Loaded " << data.size() << " samples with " 
                  << features.size() << " features." << std::endl;
        
        // Split data
        auto [train_pair, test_pair] = HealthDataLoader::train_test_split(data, targets, 0.2);
        auto& [train_data, train_targets] = train_pair;
        auto& [test_data, test_targets] = test_pair;
        
        std::cout << "Training set: " << train_data.size() << " samples" << std::endl;
        std::cout << "Test set: " << test_data.size() << " samples" << std::endl;
        
        // Define network architecture
        std::vector<unsigned> hidden_layers = {64, 32};
        
        // Create and train model
        HealthScorePredictor predictor(
            features, 
            hidden_layers,
            HealthScorePredictor::ActivationType::RELU,
            HealthScorePredictor::ActivationType::SIGMOID,
            0.01,  // learning rate
            0.9    // momentum
        );
        
        std::cout << "Training model..." << std::endl;
        double validation_error = predictor.train(
            train_data, 
            train_targets,
            32,     // batch size
            1000,   // max epochs
            20,     // early stopping patience
            0.2     // validation split
        );
        
        std::cout << "Training completed with validation error: " << validation_error << std::endl;
        
        // Evaluate model
        auto metrics = HealthDataLoader::evaluate_model(predictor, test_data, test_targets);
        
        std::cout << "Model evaluation:" << std::endl;
        for (const auto& [metric, value] : metrics) {
            std::cout << "  " << metric << ": " << value << std::endl;
        }
        
        // Save model
        predictor.save_model("health_model.dat");
        std::cout << "Model saved to health_model.dat" << std::endl;
        
        // Get feature importance
        auto importance = predictor.get_feature_importance();
        
        std::cout << "Feature importance:" << std::endl;
        for (const auto& [feature, score] : importance) {
            std::cout << "  " << feature << ": " << score << std::endl;
        }
        
        // Example inferences with diverse, realistic profiles
std::cout << "\nHealth Score Predictions for Diverse Profiles:" << std::endl;
std::cout << "============================================" << std::endl;

// Define a set of diverse, realistic profiles for inference
std::vector<std::pair<std::string, std::vector<double>>> inference_profiles = {
    {"Young Healthy Adult", {
        28.0,    // age
        1.0,     // gender (1 = female)
        62000.0, // income
        16.0,    // education_years 
        8.0,     // sleep_hours
        6.0,     // physical_activity (days/week)
        8.0,     // diet_score (0-10)
        3.0,     // stress_level (0-10, lower is better)
        7.0,     // work_life_balance (0-10)
        22.5,    // bmi
        110.0,   // systolic_bp
        70.0,    // diastolic_bp
        165.0,   // cholesterol
        60.0,    // resting_heart_rate
        1.0,     // regular_checkups (1 = yes)
        3.0,     // pollution_exposure (0-10)
        6.5,     // green_space_access (0-10)
        8.0,     // walkability_score (0-10)
        0.0,     // smoking (0-10)
        1.5,     // alcohol_consumption (drinks/week)
        0.0,     // recreational_drug_use (0-5)
        9.5,     // seat_belt_use (0-10)
        8.0,     // social_connections (0-10)
        6.0,     // community_engagement (0-10)
        2.0,     // depression_score (0-10)
        2.5,     // anxiety_score (0-10)
        0.0,     // chronic_diseases (count)
        1.0,     // family_history_risk (0-10)
        8.0,     // healthcare_access (0-10)
        7.0      // health_insurance_quality (0-10)
    }},
    {"Older Adult with Chronic Conditions", {
        68.0,    // age
        0.0,     // gender (0 = male)
        55000.0, // income
        14.0,    // education_years
        6.5,     // sleep_hours
        2.5,     // physical_activity (days/week)
        7.0,     // diet_score (0-10)
        4.5,     // stress_level (0-10, lower is better)
        6.0,     // work_life_balance (0-10)
        27.5,    // bmi
        138.0,   // systolic_bp
        85.0,    // diastolic_bp
        210.0,   // cholesterol
        74.0,    // resting_heart_rate
        1.0,     // regular_checkups (1 = yes)
        3.5,     // pollution_exposure (0-10)
        5.0,     // green_space_access (0-10)
        5.0,     // walkability_score (0-10)
        0.0,     // smoking (0-10)
        1.0,     // alcohol_consumption (drinks/week)
        0.0,     // recreational_drug_use (0-5)
        9.0,     // seat_belt_use (0-10)
        7.0,     // social_connections (0-10)
        6.0,     // community_engagement (0-10)
        3.0,     // depression_score (0-10)
        3.0,     // anxiety_score (0-10)
        2.5,     // chronic_diseases (count)
        4.0,     // family_history_risk (0-10)
        8.0,     // healthcare_access (0-10)
        7.0      // health_insurance_quality (0-10)
    }},
    {"Sedentary Smoker", {
        42.0,    // age
        0.0,     // gender (0 = male)
        48000.0, // income
        12.0,    // education_years
        5.5,     // sleep_hours
        1.0,     // physical_activity (days/week)
        4.0,     // diet_score (0-10)
        7.0,     // stress_level (0-10, lower is better)
        4.0,     // work_life_balance (0-10)
        31.0,    // bmi
        142.0,   // systolic_bp
        92.0,    // diastolic_bp
        225.0,   // cholesterol
        82.0,    // resting_heart_rate
        0.0,     // regular_checkups (0 = no)
        6.0,     // pollution_exposure (0-10)
        3.0,     // green_space_access (0-10)
        4.0,     // walkability_score (0-10)
        7.5,     // smoking (0-10)
        5.0,     // alcohol_consumption (drinks/week)
        0.5,     // recreational_drug_use (0-5)
        6.0,     // seat_belt_use (0-10)
        4.0,     // social_connections (0-10)
        2.0,     // community_engagement (0-10)
        5.0,     // depression_score (0-10)
        6.0,     // anxiety_score (0-10)
        1.0,     // chronic_diseases (count)
        3.0,     // family_history_risk (0-10)
        5.0,     // healthcare_access (0-10)
        4.0      // health_insurance_quality (0-10)
    }},
    {"Low-Income Student", {
        22.0,    // age
        1.0,     // gender (1 = female)
        18000.0, // income
        14.0,    // education_years (in progress)
        6.0,     // sleep_hours
        3.5,     // physical_activity (days/week)
        5.0,     // diet_score (0-10)
        8.0,     // stress_level (0-10, lower is better)
        5.0,     // work_life_balance (0-10)
        23.0,    // bmi
        118.0,   // systolic_bp
        75.0,    // diastolic_bp
        170.0,   // cholesterol
        72.0,    // resting_heart_rate
        0.0,     // regular_checkups (0 = no)
        4.0,     // pollution_exposure (0-10)
        5.0,     // green_space_access (0-10)
        7.0,     // walkability_score (0-10)
        2.0,     // smoking (0-10)
        3.0,     // alcohol_consumption (drinks/week)
        0.5,     // recreational_drug_use (0-5)
        8.0,     // seat_belt_use (0-10)
        7.0,     // social_connections (0-10)
        4.0,     // community_engagement (0-10)
        4.0,     // depression_score (0-10)
        5.0,     // anxiety_score (0-10)
        0.0,     // chronic_diseases (count)
        2.0,     // family_history_risk (0-10)
        4.0,     // healthcare_access (0-10)
        3.0      // health_insurance_quality (0-10)
    }},
    {"Health-Conscious Professional", {
        38.0,    // age
        1.0,     // gender (1 = female)
        85000.0, // income
        18.0,    // education_years
        7.5,     // sleep_hours
        5.0,     // physical_activity (days/week)
        9.0,     // diet_score (0-10)
        4.0,     // stress_level (0-10, lower is better)
        7.0,     // work_life_balance (0-10)
        23.0,    // bmi
        116.0,   // systolic_bp
        74.0,    // diastolic_bp
        165.0,   // cholesterol
        64.0,    // resting_heart_rate
        1.0,     // regular_checkups (1 = yes)
        3.0,     // pollution_exposure (0-10)
        7.0,     // green_space_access (0-10)
        7.0,     // walkability_score (0-10)
        0.0,     // smoking (0-10)
        2.0,     // alcohol_consumption (drinks/week)
        0.0,     // recreational_drug_use (0-5)
        10.0,    // seat_belt_use (0-10)
        8.0,     // social_connections (0-10)
        6.0,     // community_engagement (0-10)
        2.0,     // depression_score (0-10)
        3.0,     // anxiety_score (0-10)
        0.0,     // chronic_diseases (count)
        2.0,     // family_history_risk (0-10)
        9.0,     // healthcare_access (0-10)
        8.0      // health_insurance_quality (0-10)
    }}
};

// Make predictions for each profile
for (const auto& [profile_name, profile_values] : inference_profiles) {
    std::cout << "\nProfile: " << profile_name << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    
    // Display key health metrics
    std::cout << "Key metrics:" << std::endl;
    std::vector<std::pair<std::string, int>> key_metrics = {
        {"Age", 0}, {"BMI", 9}, {"Physical Activity", 5}, 
        {"Diet Score", 6}, {"Stress Level", 7}, {"Smoking", 18},
        {"Chronic Diseases", 26}
    };
    
    for (const auto& [metric, idx] : key_metrics) {
        std::cout << "  " << metric << ": " << profile_values[idx] << std::endl;
    }
    
    // Make prediction
    double health_score = predictor.predict(profile_values);
    std::cout << "\nPredicted health score: " << health_score << std::endl;
    
    // Explain prediction
    auto explanations = predictor.explain_prediction(profile_values);
    
    // Show top 3 positive contributors
    std::cout << "\nTop positive contributors:" << std::endl;
    int pos_count = 0;
    for (const auto& [feature, contribution] : explanations) {
        if (contribution > 0) {
            std::cout << "  " << feature << ": +" << contribution << std::endl;
            if (++pos_count >= 3) break;
        }
    }
    
    // Show top 3 negative contributors
    std::cout << "\nTop negative contributors:" << std::endl;
    int neg_count = 0;
    for (const auto& [feature, contribution] : explanations) {
        if (contribution < 0) {
            std::cout << "  " << feature << ": " << contribution << std::endl;
            if (++neg_count >= 3) break;
        }
    }
    
    std::cout << "------------------------------------------" << std::endl;
}
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}