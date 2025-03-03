/**
 * @file NeuralNetwork.hpp
 * @brief Feed-forward neural network with one hidden layer
 * @details Used for Q-function approximation. Supports forward pass with ReLU
 *          activation and training via backpropagation with gradient clipping.
 *
 * Time Complexity: O(inputSize * hiddenSize + hiddenSize * outputSize) per forward/train
 * Space Complexity: O(inputSize * hiddenSize + hiddenSize * outputSize) for weights
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @class NeuralNetwork
 * @brief Feed-forward neural network with one hidden layer
 *
 * Used for Q-function approximation. Supports forward pass with ReLU activation
 * and training via backpropagation with gradient clipping.
 */
class NeuralNetwork {
public:
    /**
     * @brief Construct a neural network
     * @param inputSize  Number of input neurons (state dimension for Q-learning)
     * @param hiddenSize Number of hidden-layer neurons — controls the network's
     *                   capacity to approximate the Q-function.  More neurons
     *                   can model more complex state-action mappings but increase
     *                   training time.  Typical values: 8-32.
     * @param outputSize Number of output neurons (one Q-value per action dimension)
     * @param learningRate Step size for gradient descent weight updates
     *
     * Weights are initialized with Xavier/Glorot scaling:
     *   w ~ N(0, 0.1) * sqrt(2 / (fan_in + fan_out))
     * This keeps activation variances roughly constant across layers,
     * preventing gradients from vanishing or exploding at the start
     * of training.
     */
    NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize,
                  double learningRate = 0.001);

    /**
     * @brief Forward pass through the network
     * @param input Input vector (must match inputSize)
     * @return Output vector
     * @throws std::invalid_argument if input size mismatches
     *
     * Time Complexity: O(inputSize * hiddenSize + hiddenSize * outputSize)
     */
    std::vector<double> Forward(const std::vector<double>& input);

    /**
     * @brief Train via backpropagation (single sample)
     * @param input Input vector
     * @param target Target output vector
     * @return MSE loss
     *
     * Time Complexity: O(inputSize * hiddenSize + hiddenSize * outputSize)
     */
    double Train(const std::vector<double>& input, const std::vector<double>& target);

    /**
     * @brief Save model weights to file
     * @param filename Output file path
     */
    void SaveModel(const std::string& filename);

    /**
     * @brief Load model weights from file
     * @param filename Input file path
     * @throws std::runtime_error if file cannot be opened
     */
    void LoadModel(const std::string& filename);

    /** @return Number of input neurons */
    size_t GetInputSize() const { return inputSize_; }
    /** @return Number of hidden neurons */
    size_t GetHiddenSize() const { return hiddenSize_; }
    /** @return Number of output neurons */
    size_t GetOutputSize() const { return outputSize_; }

private:
    size_t inputSize_;   ///< State dimension (e.g. 3: error, error_deriv, control)
    size_t hiddenSize_;  ///< Hidden-layer width — capacity of Q-function approximator
    size_t outputSize_;  ///< Action dimension (e.g. 3: one Q-value per PID gain)
    double learningRate_;
    std::vector<std::vector<double>> w1_, w2_;
    std::vector<double> b1_, b2_, hidden_;
    mutable std::mutex mutex_;
};
