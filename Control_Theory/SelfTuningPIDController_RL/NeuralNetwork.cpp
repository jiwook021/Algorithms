/**
 * @file NeuralNetwork.cpp
 * @brief Implementation of the NeuralNetwork class
 *
 * A simple feed-forward network used as a Q-function approximator.
 * Architecture: input → hidden (ReLU) → output (clamped to [-1, 1]).
 *
 * The network maps environment states to Q-values — one per action
 * dimension.  The RL agent uses these Q-values to decide which PID
 * gains [Kp, Ki, Kd] to try next.
 *
 * Training uses single-sample backpropagation with gradient clipping
 * to keep updates stable under the non-stationary targets typical of
 * Q-learning.
 */

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize,
                             double learningRate)
    : inputSize_(inputSize), hiddenSize_(hiddenSize), outputSize_(outputSize),
      learningRate_(learningRate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);

    // Xavier/Glorot scaling: sqrt(2 / (fan_in + fan_out))
    // Keeps activation variance roughly constant across layers so
    // gradients neither vanish nor explode at the start of training.
    double inputHiddenScale  = std::sqrt(2.0 / (inputSize_ + hiddenSize_));
    double hiddenOutputScale = std::sqrt(2.0 / (hiddenSize_ + outputSize_));

    // W1: input → hidden  (inputSize x hiddenSize matrix)
    w1_.resize(inputSize_, std::vector<double>(hiddenSize_));
    for (size_t i = 0; i < inputSize_; ++i)
        for (size_t j = 0; j < hiddenSize_; ++j)
            w1_[i][j] = dist(gen) * inputHiddenScale;

    // W2: hidden → output (hiddenSize x outputSize matrix)
    w2_.resize(hiddenSize_, std::vector<double>(outputSize_));
    for (size_t i = 0; i < hiddenSize_; ++i)
        for (size_t j = 0; j < outputSize_; ++j)
            w2_[i][j] = dist(gen) * hiddenOutputScale;

    b1_.resize(hiddenSize_, 0.0);
    b2_.resize(outputSize_, 0.0);
}

std::vector<double> NeuralNetwork::Forward(const std::vector<double>& input) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (input.size() != inputSize_) {
        throw std::invalid_argument("Input size mismatch: expected " +
                                     std::to_string(inputSize_) + ", got " +
                                     std::to_string(input.size()));
    }

    // Clamp inputs to [-10, 10] to prevent extreme activations
    std::vector<double> scaledInput = input;
    constexpr double MAX_INPUT_VAL = 10.0;
    for (auto& v : scaledInput) v = std::clamp(v, -MAX_INPUT_VAL, MAX_INPUT_VAL);

    // Hidden layer: linear transform + ReLU activation
    //   h_j = max(0, b1_j + sum_i(input_i * W1_ij))
    hidden_.resize(hiddenSize_);
    for (size_t j = 0; j < hiddenSize_; ++j) {
        hidden_[j] = b1_[j];
        for (size_t i = 0; i < inputSize_; ++i)
            hidden_[j] += scaledInput[i] * w1_[i][j];
        hidden_[j] = std::max(0.0, hidden_[j]);
    }

    // Output layer: linear transform, clamped to [-1, 1]
    // The RL agent rescales these to actual PID gain ranges
    std::vector<double> output(outputSize_);
    for (size_t k = 0; k < outputSize_; ++k) {
        output[k] = b2_[k];
        for (size_t j = 0; j < hiddenSize_; ++j)
            output[k] += hidden_[j] * w2_[j][k];
        output[k] = std::clamp(output[k], -1.0, 1.0);
    }
    return output;
}

double NeuralNetwork::Train(const std::vector<double>& input, const std::vector<double>& target) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (input.size() != inputSize_ || target.size() != outputSize_) {
        throw std::invalid_argument("Size mismatch in Train");
    }

    // Skip update if inputs contain NaN/Inf — prevents weight corruption
    for (const auto& v : input)
        if (std::isnan(v) || std::isinf(v)) return 0.0;
    for (const auto& v : target)
        if (std::isnan(v) || std::isinf(v)) return 0.0;

    // ── Forward pass (same as Forward(), but keeps local activations
    //    for the backward pass) ──

    std::vector<double> scaledInput = input;
    constexpr double MAX_INPUT_VAL = 10.0;
    for (auto& v : scaledInput) v = std::clamp(v, -MAX_INPUT_VAL, MAX_INPUT_VAL);

    std::vector<double> hiddenActivations(hiddenSize_);
    for (size_t j = 0; j < hiddenSize_; ++j) {
        hiddenActivations[j] = b1_[j];
        for (size_t i = 0; i < inputSize_; ++i)
            hiddenActivations[j] += scaledInput[i] * w1_[i][j];
        hiddenActivations[j] = std::max(0.0, hiddenActivations[j]);
    }

    std::vector<double> output(outputSize_);
    for (size_t k = 0; k < outputSize_; ++k) {
        output[k] = b2_[k];
        for (size_t j = 0; j < hiddenSize_; ++j)
            output[k] += hiddenActivations[j] * w2_[j][k];
        output[k] = std::clamp(output[k], -1.0, 1.0);
    }

    // ── MSE loss: mean of (target_k - output_k)^2 ──
    double loss = 0.0;
    for (size_t k = 0; k < outputSize_; ++k) {
        double err = target[k] - output[k];
        loss += err * err;
    }
    loss /= outputSize_;

    // ── Backward pass ──
    // Gradient clipping prevents any single update from destabilizing
    // the network — important because Q-learning targets shift every step.
    constexpr double MAX_GRAD   = 1.0;   // clip per-sample gradient
    constexpr double MAX_UPDATE = 0.1;   // clip weight update magnitude

    // Output-layer error gradient: dLoss/dOutput_k = 2*(target-output)/N
    std::vector<double> outputGrad(outputSize_);
    for (size_t k = 0; k < outputSize_; ++k)
        outputGrad[k] = std::clamp((target[k] - output[k]) * 2.0 / static_cast<double>(outputSize_),
                                     -MAX_GRAD, MAX_GRAD);

    // Update W2 (hidden→output weights) and B2 (output biases)
    for (size_t j = 0; j < hiddenSize_; ++j)
        for (size_t k = 0; k < outputSize_; ++k)
            w2_[j][k] += std::clamp(learningRate_ * hiddenActivations[j] * outputGrad[k],
                                     -MAX_UPDATE, MAX_UPDATE);
    for (size_t k = 0; k < outputSize_; ++k)
        b2_[k] += std::clamp(learningRate_ * outputGrad[k], -MAX_UPDATE, MAX_UPDATE);

    // Hidden-layer gradient: backprop through ReLU (gradient is 0 if activation was 0)
    std::vector<double> hiddenGrad(hiddenSize_, 0.0);
    for (size_t j = 0; j < hiddenSize_; ++j) {
        for (size_t k = 0; k < outputSize_; ++k)
            hiddenGrad[j] += outputGrad[k] * w2_[j][k];
        hiddenGrad[j] *= (hiddenActivations[j] > 0.0) ? 1.0 : 0.0;
        hiddenGrad[j] = std::clamp(hiddenGrad[j], -MAX_GRAD, MAX_GRAD);
    }

    // Update W1 (input→hidden weights) and B1 (hidden biases)
    for (size_t i = 0; i < inputSize_; ++i)
        for (size_t j = 0; j < hiddenSize_; ++j)
            w1_[i][j] += std::clamp(learningRate_ * scaledInput[i] * hiddenGrad[j],
                                     -MAX_UPDATE, MAX_UPDATE);
    for (size_t j = 0; j < hiddenSize_; ++j)
        b1_[j] += std::clamp(learningRate_ * hiddenGrad[j], -MAX_UPDATE, MAX_UPDATE);

    return loss;
}

void NeuralNetwork::SaveModel(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file for saving: " + filename);

    file << inputSize_ << " " << hiddenSize_ << " " << outputSize_ << " " << learningRate_ << "\n";
    for (size_t i = 0; i < inputSize_; ++i) {
        for (size_t j = 0; j < hiddenSize_; ++j) file << w1_[i][j] << " ";
        file << "\n";
    }
    for (size_t i = 0; i < hiddenSize_; ++i) {
        for (size_t j = 0; j < outputSize_; ++j) file << w2_[i][j] << " ";
        file << "\n";
    }
    for (size_t i = 0; i < hiddenSize_; ++i) file << b1_[i] << " ";
    file << "\n";
    for (size_t i = 0; i < outputSize_; ++i) file << b2_[i] << " ";
    file.close();
}

void NeuralNetwork::LoadModel(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file for loading: " + filename);

    file >> inputSize_ >> hiddenSize_ >> outputSize_ >> learningRate_;
    w1_.resize(inputSize_, std::vector<double>(hiddenSize_));
    w2_.resize(hiddenSize_, std::vector<double>(outputSize_));
    b1_.resize(hiddenSize_);
    b2_.resize(outputSize_);
    hidden_.resize(hiddenSize_);

    for (size_t i = 0; i < inputSize_; ++i)
        for (size_t j = 0; j < hiddenSize_; ++j) file >> w1_[i][j];
    for (size_t i = 0; i < hiddenSize_; ++i)
        for (size_t j = 0; j < outputSize_; ++j) file >> w2_[i][j];
    for (size_t i = 0; i < hiddenSize_; ++i) file >> b1_[i];
    for (size_t i = 0; i < outputSize_; ++i) file >> b2_[i];
    file.close();
}
