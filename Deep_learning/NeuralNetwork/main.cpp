#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <sstream>
#include <functional>
#include <numeric>
#include <string>

// Forward declarations
class Layer;
class NeuralNetwork;

/**
 * @brief Base class for activation functions
 */
class ActivationFunction {
public:
    virtual double activate(double x) const = 0;
    virtual double derivative(double x) const = 0;
    virtual std::string name() const = 0;
    virtual ~ActivationFunction() = default;
};

/**
 * @brief Sigmoid activation function
 * f(x) = 1 / (1 + e^(-x))
 * f'(x) = f(x) * (1 - f(x))
 */
class Sigmoid : public ActivationFunction {
public:
    double activate(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }
    
    double derivative(double x) const override {
        double sig = activate(x);
        return sig * (1.0 - sig);
    }
    
    std::string name() const override {
        return "Sigmoid";
    }
};

/**
 * @brief ReLU activation function
 * f(x) = max(0, x)
 * f'(x) = 1 if x > 0, 0 otherwise
 */
class ReLU : public ActivationFunction {
public:
    double activate(double x) const override {
        return std::max(0.0, x);
    }
    
    double derivative(double x) const override {
        return x > 0 ? 1.0 : 0.0;
    }
    
    std::string name() const override {
        return "ReLU";
    }
};

/**
 * @brief Tanh activation function
 * f(x) = tanh(x)
 * f'(x) = 1 - tanh^2(x)
 */
class Tanh : public ActivationFunction {
public:
    double activate(double x) const override {
        return std::tanh(x);
    }
    
    double derivative(double x) const override {
        double th = std::tanh(x);
        return 1.0 - th * th;
    }
    
    std::string name() const override {
        return "Tanh";
    }
};

/**
 * @brief Neural network layer
 * Represents a single layer in the neural network with weights, biases,
 * and activation functions.
 */
class Layer {
public:
    /**
     * @brief Construct a new Layer
     * 
     * @param inputSize Number of inputs to this layer
     * @param outputSize Number of neurons in this layer
     * @param activation Activation function to use
     * @throws std::invalid_argument if input or output size is invalid
     */
    Layer(size_t inputSize, size_t outputSize, std::unique_ptr<ActivationFunction> activation) 
        : inputSize(inputSize), 
          outputSize(outputSize),
          activationFunc(std::move(activation)),
          weights(outputSize, std::vector<double>(inputSize)),
          biases(outputSize),
          outputs(outputSize),
          rawInputs(outputSize),
          deltas(outputSize) {
        
        if (inputSize == 0 || outputSize == 0) {
            throw std::invalid_argument("Layer sizes must be greater than zero");
        }
        
        // Initialize weights and biases with Xavier/Glorot initialization
        // This helps with training convergence
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = std::sqrt(6.0 / (inputSize + outputSize));
        std::uniform_real_distribution<double> dist(-limit, limit);
        
        for (auto& neuronWeights : weights) {
            for (auto& weight : neuronWeights) {
                weight = dist(gen);
            }
        }
        
        for (auto& bias : biases) {
            bias = dist(gen);
        }
    }
    
    /**
     * @brief Forward pass through the layer
     * 
     * @param inputs Input values to the layer
     * @return std::vector<double> Output values from the layer
     * @throws std::invalid_argument if inputs size doesn't match expected size
     */
    std::vector<double> forward(const std::vector<double>& inputs) {
        if (inputs.size() != inputSize) {
            std::stringstream ss;
            ss << "Input size mismatch: expected " << inputSize << ", got " << inputs.size();
            throw std::invalid_argument(ss.str());
        }
        
        // Calculate weighted sum and apply activation function
        // z = weights * inputs + bias
        // a = activation(z)
        for (size_t i = 0; i < outputSize; ++i) {
            double sum = biases[i];
            for (size_t j = 0; j < inputSize; ++j) {
                sum += weights[i][j] * inputs[j];
            }
            // Store both raw input (for backprop) and activated output
            rawInputs[i] = sum;
            outputs[i] = activationFunc->activate(sum);
        }
        
        return outputs;
    }
    
    /**
     * @brief Backward pass for output layer
     * 
     * @param expected Expected output values
     * @return std::vector<double> Delta values for this layer
     * @throws std::invalid_argument if expected size doesn't match layer size
     */
    std::vector<double> backwardOutput(const std::vector<double>& expected) {
        if (expected.size() != outputSize) {
            std::stringstream ss;
            ss << "Expected output size mismatch: expected " << outputSize << ", got " << expected.size();
            throw std::invalid_argument(ss.str());
        }
        
        // Calculate deltas for output layer
        // delta = (output - expected) * activation'(rawInput)
        for (size_t i = 0; i < outputSize; ++i) {
            double error = outputs[i] - expected[i];
            deltas[i] = error * activationFunc->derivative(rawInputs[i]);
        }
        
        return deltas;
    }
    
    /**
     * @brief Backward pass for hidden layer
     * 
     * @param nextLayer Next layer in the network
     * @return std::vector<double> Delta values for this layer
     */
    std::vector<double> backwardHidden(const Layer& nextLayer) {
        // Calculate deltas for hidden layer
        // delta = sum(next_deltas * next_weights) * activation'(rawInput)
        for (size_t i = 0; i < outputSize; ++i) {
            double error = 0.0;
            for (size_t j = 0; j < nextLayer.outputSize; ++j) {
                error += nextLayer.deltas[j] * nextLayer.weights[j][i];
            }
            deltas[i] = error * activationFunc->derivative(rawInputs[i]);
        }
        
        return deltas;
    }
    
    /**
     * @brief Update weights and biases
     * 
     * @param inputs Inputs to this layer
     * @param learningRate Learning rate for updates
     * @throws std::invalid_argument if inputs size doesn't match expected size
     */
    void updateWeights(const std::vector<double>& inputs, double learningRate) {
        if (inputs.size() != inputSize) {
            std::stringstream ss;
            ss << "Input size mismatch for weight update: expected " << inputSize << ", got " << inputs.size();
            throw std::invalid_argument(ss.str());
        }
        
        // Update weights and biases
        // weight = weight - learning_rate * delta * input
        // bias = bias - learning_rate * delta
        for (size_t i = 0; i < outputSize; ++i) {
            for (size_t j = 0; j < inputSize; ++j) {
                weights[i][j] -= learningRate * deltas[i] * inputs[j];
            }
            biases[i] -= learningRate * deltas[i];
        }
    }
    
    // Getters
    size_t getInputSize() const { return inputSize; }
    size_t getOutputSize() const { return outputSize; }
    const std::vector<double>& getOutputs() const { return outputs; }
    std::string getActivationName() const { return activationFunc->name(); }
    
private:
    size_t inputSize;
    size_t outputSize;
    std::unique_ptr<ActivationFunction> activationFunc;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> outputs;
    std::vector<double> rawInputs;
    std::vector<double> deltas;
    
    friend class NeuralNetwork;
};

/**
 * @brief Neural Network class
 * Implements a feedforward neural network with backpropagation
 */
class NeuralNetwork {
public:
    /**
     * @brief Construct a new Neural Network
     * 
     * @param inputSize Number of input features
     * @param learningRate Learning rate for training
     */
    NeuralNetwork(size_t inputSize, double learningRate = 0.1)
        : inputSize(inputSize), learningRate(learningRate) {
        
        if (inputSize == 0) {
            throw std::invalid_argument("Input size must be greater than zero");
        }
        
        if (learningRate <= 0.0 || learningRate > 1.0) {
            throw std::invalid_argument("Learning rate must be between 0 and 1");
        }
    }
    
    /**
     * @brief Add a layer to the network
     * 
     * @param neurons Number of neurons in the layer
     * @param activation Activation function to use
     * @throws std::invalid_argument if neuron count is invalid
     */
    void addLayer(size_t neurons, std::unique_ptr<ActivationFunction> activation) {
        if (neurons == 0) {
            throw std::invalid_argument("Number of neurons must be greater than zero");
        }
        
        size_t prevSize = layers.empty() ? inputSize : layers.back().getOutputSize();
        layers.emplace_back(prevSize, neurons, std::move(activation));
    }
    
    // Convenience functions for common activation functions
    void addSigmoidLayer(size_t neurons) {
        addLayer(neurons, std::make_unique<Sigmoid>());
    }
    
    void addReLULayer(size_t neurons) {
        addLayer(neurons, std::make_unique<ReLU>());
    }
    
    void addTanhLayer(size_t neurons) {
        addLayer(neurons, std::make_unique<Tanh>());
    }
    
    /**
     * @brief Set the learning rate
     * 
     * @param rate New learning rate
     * @throws std::invalid_argument if rate is invalid
     */
    void setLearningRate(double rate) {
        std::unique_lock<std::shared_mutex> lock(mutex); // Exclusive lock for writing
        
        if (rate <= 0.0 || rate > 1.0) {
            throw std::invalid_argument("Learning rate must be between 0 and 1");
        }
        
        learningRate = rate;
    }
    
    /**
     * @brief Get the current learning rate
     * 
     * @return double Current learning rate
     */
    double getLearningRate() const {
        std::shared_lock<std::shared_mutex> lock(mutex); // Shared lock for reading
        return learningRate;
    }
    
    /**
     * @brief Forward pass through the network
     * 
     * @param inputs Input values to the network
     * @return std::vector<double> Output values from the network
     * @throws std::invalid_argument if inputs size doesn't match expected size
     * @throws std::runtime_error if network has no layers
     */
    std::vector<double> forward(const std::vector<double>& inputs) {
        std::shared_lock<std::shared_mutex> lock(mutex); // Shared lock for reading network state
        
        if (layers.empty()) {
            throw std::runtime_error("Network has no layers");
        }
        
        if (inputs.size() != inputSize) {
            std::stringstream ss;
            ss << "Input size mismatch: expected " << inputSize << ", got " << inputs.size();
            throw std::invalid_argument(ss.str());
        }
        
        // Pass inputs through each layer
        std::vector<double> layerInputs = inputs;
        for (auto& layer : layers) {
            layerInputs = layer.forward(layerInputs);
        }
        
        return layerInputs; // Final layer outputs
    }
    
    /**
     * @brief Train the network with a single sample
     * 
     * @param inputs Input values
     * @param expected Expected output values
     * @return double Mean squared error
     * @throws std::invalid_argument if input or expected size doesn't match
     * @throws std::runtime_error if network has no layers
     */
    double train(const std::vector<double>& inputs, const std::vector<double>& expected) {
        std::unique_lock<std::shared_mutex> lock(mutex); // Exclusive lock for writing network state
        
        if (layers.empty()) {
            throw std::runtime_error("Network has no layers");
        }
        
        if (inputs.size() != inputSize) {
            std::stringstream ss;
            ss << "Input size mismatch: expected " << inputSize << ", got " << inputs.size();
            throw std::invalid_argument(ss.str());
        }
        
        if (expected.size() != layers.back().getOutputSize()) {
            std::stringstream ss;
            ss << "Expected output size mismatch: expected " << layers.back().getOutputSize() 
                << ", got " << expected.size();
            throw std::invalid_argument(ss.str());
        }
        
        // Forward pass
        std::vector<std::vector<double>> layerInputs{inputs};
        std::vector<double> currentInputs = inputs;
        
        for (auto& layer : layers) {
            currentInputs = layer.forward(currentInputs);
            layerInputs.push_back(currentInputs);
        }
        
        // Calculate output error
        double error = 0.0;
        const auto& outputs = layers.back().getOutputs();
        for (size_t i = 0; i < expected.size(); ++i) {
            double diff = outputs[i] - expected[i];
            error += diff * diff;
        }
        error /= expected.size();
        
        // Backward pass
        // Output layer
        layers.back().backwardOutput(expected);
        
        // Hidden layers
        for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i) {
            layers[i].backwardHidden(layers[i + 1]);
        }
        
        // Update weights
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].updateWeights(layerInputs[i], learningRate);
        }
        
        return error;
    }
    
    /**
     * @brief Train the network with a batch of samples
     * 
     * @param inputs Vector of input samples
     * @param expected Vector of expected outputs
     * @param epochs Number of training epochs
     * @param batchSize Size of mini-batches
     * @param useThreads Whether to use parallel processing for batches
     * @return std::vector<double> Error history
     * @throws std::invalid_argument if input params are invalid
     */
    std::vector<double> trainBatch(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& expected,
        size_t epochs = 1000,
        size_t batchSize = 32,
        bool useThreads = true
    ) {
        if (inputs.size() != expected.size() || inputs.empty()) {
            throw std::invalid_argument("Input and expected output sizes must match and be non-empty");
        }
        
        if (epochs == 0) {
            throw std::invalid_argument("Number of epochs must be greater than zero");
        }
        
        if (batchSize == 0 || batchSize > inputs.size()) {
            batchSize = inputs.size();
        }
        
        std::vector<double> errorHistory;
        errorHistory.reserve(epochs);
        
        // Prepare indices for shuffling
        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            // Shuffle indices
            std::shuffle(indices.begin(), indices.end(), gen);
            
            double epochError = 0.0;
            
            // Process mini-batches
            for (size_t batchStart = 0; batchStart < inputs.size(); batchStart += batchSize) {
                size_t currentBatchSize = std::min(batchSize, inputs.size() - batchStart);
                
                if (useThreads && currentBatchSize > 1) {
                    // Parallel processing of the batch
                    std::vector<std::future<double>> futures;
                    futures.reserve(currentBatchSize);
                    
                    for (size_t i = 0; i < currentBatchSize; ++i) {
                        size_t idx = indices[batchStart + i];
                        futures.push_back(std::async(std::launch::async, [this, &inputs, &expected, idx]() {
                            return this->train(inputs[idx], expected[idx]);
                        }));
                    }
                    
                    // Collect errors
                    for (auto& future : futures) {
                        epochError += future.get();
                    }
                } else {
                    // Sequential processing
                    for (size_t i = 0; i < currentBatchSize; ++i) {
                        size_t idx = indices[batchStart + i];
                        epochError += train(inputs[idx], expected[idx]);
                    }
                }
            }
            
            // Calculate average error for epoch
            epochError /= inputs.size();
            errorHistory.push_back(epochError);
        }
        
        return errorHistory;
    }
    
    /**
     * @brief Print network architecture
     */
    void printArchitecture() const {
        std::shared_lock<std::shared_mutex> lock(mutex); // Shared lock for reading
        
        std::cout << "Neural Network Architecture:\n";
        std::cout << "Input size: " << inputSize << "\n";
        
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            std::cout << "Layer " << (i + 1) << ": "
                      << layer.getInputSize() << " -> " 
                      << layer.getOutputSize() << ", "
                      << "Activation: " << layer.getActivationName() << "\n";
        }
        
        std::cout << "Learning rate: " << learningRate << "\n";
    }
    
    /**
     * @brief Save network to file (stub implementation)
     * 
     * @param filename File to save to
     * @return bool Success status
     */
    bool save(const std::string& filename) const {
        std::shared_lock<std::shared_mutex> lock(mutex); // Shared lock for reading
        
        // Actual implementation would save weights, biases, and architecture
        std::cout << "Saving network to " << filename << " (not implemented)\n";
        return true;
    }
    
    /**
     * @brief Load network from file (stub implementation)
     * 
     * @param filename File to load from
     * @return bool Success status
     */
    bool load(const std::string& filename) {
        std::unique_lock<std::shared_mutex> lock(mutex); // Exclusive lock for writing
        
        // Actual implementation would load weights, biases, and architecture
        std::cout << "Loading network from " << filename << " (not implemented)\n";
        return true;
    }
    
private:
    size_t inputSize;
    double learningRate;
    std::vector<Layer> layers;
    mutable std::shared_mutex mutex; // For thread safety
};

/**
 * @brief Test function to demonstrate XOR problem
 */
void testXOR() {
    std::cout << "=== XOR Problem Test ===\n";
    
    // Create inputs and expected outputs for XOR
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    
    std::vector<std::vector<double>> outputs = {
        {0}, {1}, {1}, {0}
    };
    
    // Create network
    NeuralNetwork nn(2, 0.1);
    nn.addSigmoidLayer(4);
    nn.addSigmoidLayer(1);
    
    nn.printArchitecture();
    
    // Train network
    std::cout << "Training...\n";
    auto errors = nn.trainBatch(inputs, outputs, 10000, 4);
    
    // Print final error
    std::cout << "Final error: " << errors.back() << "\n";
    
    // Test network
    std::cout << "Testing...\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto result = nn.forward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] 
                  << "] Output: " << result[0] 
                  << " Expected: " << outputs[i][0] << "\n";
    }
    
    std::cout << "=== End XOR Test ===\n\n";
}

/**
 * @brief Main function
 */
int main() {
    try {
        testXOR();
        
        // Additional test with MNIST would go here in a real implementation
        // testMNIST();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

/*
Time and Space Complexity Analysis:

Time Complexity:
- Forward pass: O(W) where W is the total number of weights in the network
- Backward pass: O(W) where W is the total number of weights in the network
- Batch training: O(E * N * W) where E is epochs, N is number of samples, W is weights

Space Complexity:
- Network storage: O(W) where W is the total number of weights
- Training: O(L + B) where L is the number of layers and B is the batch size

Alternative Implementations and Trade-offs:

1. Using Eigen or other matrix libraries:
   - Pro: Faster computation, especially on large networks
   - Pro: Potential GPU acceleration
   - Con: External dependency
   - Con: More complex code

2. Using convolutional layers (for image recognition):
   - Pro: Much better for image recognition tasks
   - Pro: Reduced parameter count for same capacity
   - Con: More complex implementation
   - Con: Higher computational requirements

3. Using LSTM or GRU layers (for sequence data):
   - Pro: Better for sequential data like text
   - Pro: Can maintain state between predictions
   - Con: Much more complex implementation
   - Con: Harder to train effectively

4. Optimizers beyond basic gradient descent:
   - Pro: Faster convergence and better results
   - Pro: Less sensitivity to learning rate
   - Con: More complex implementation
   - Con: More hyperparameters to tune

Security Considerations:
- Validate all inputs to prevent buffer overflows
- Avoid using untrusted data in file paths when saving/loading
- Consider timing attacks when using neural networks for authentication
- Be aware of potential data leakage through inference attacks
*/