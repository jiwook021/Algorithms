/**
 * @file NeuralNetwork.hpp
 * @brief Feedforward neural network with backpropagation
 * @details Implements a multi-layer perceptron with configurable activation functions
 *          (Sigmoid, ReLU, Tanh), Xavier initialization, and mini-batch training.
 *          Thread-safe for concurrent forward/train calls via shared_mutex.
 *
 */
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <functional>
#include <stdexcept>
#include <iostream>

namespace Ml {

// =====================================================================
// Abstract base class (runtime polymorphism for heterogeneous layers)
// =====================================================================

/**
 * @brief Base class for activation functions.
 */
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual double Activate(double x) const = 0;
    virtual double Derivative(double x) const = 0;
    virtual std::string Name() const = 0;
};

/**
 * @brief Sigmoid: f(x) = 1 / (1 + e^(-x))
 */
class Sigmoid : public ActivationFunction {
public:
    double Activate(double x) const override {
        return 1.0 / (1.0 + std::exp(-x));
    }
    double Derivative(double x) const override {
        double s = Activate(x);
        return s * (1.0 - s);
    }
    std::string Name() const override { return "Sigmoid"; }
};

/**
 * @brief ReLU: f(x) = max(0, x)
 */
class ReLU : public ActivationFunction {
public:
    double Activate(double x) const override { return std::max(0.0, x); }
    double Derivative(double x) const override { return x > 0 ? 1.0 : 0.0; }
    std::string Name() const override { return "ReLU"; }
};

/**
 * @brief Tanh: f(x) = tanh(x)
 */
class Tanh : public ActivationFunction {
public:
    double Activate(double x) const override { return std::tanh(x); }
    double Derivative(double x) const override {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }
    std::string Name() const override { return "Tanh"; }
};

// Forward declaration
class NeuralNetwork;

/**
 * @brief A single dense layer with weights, biases, and activation.
 */
class Layer {
public:
    Layer(size_t input_size, size_t output_size,
          std::unique_ptr<ActivationFunction> activation);

    std::vector<double> Forward(const std::vector<double>& inputs);
    std::vector<double> BackwardOutput(const std::vector<double>& expected);
    std::vector<double> BackwardHidden(const Layer& next_layer);
    void UpdateWeights(const std::vector<double>& inputs, double learning_rate);

    size_t GetInputSize() const { return input_size_; }
    size_t GetOutputSize() const { return output_size_; }
    const std::vector<double>& GetOutputs() const { return outputs_; }
    std::string GetActivationName() const { return activation_func_->Name(); }

private:
    size_t input_size_;
    size_t output_size_;
    std::unique_ptr<ActivationFunction> activation_func_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;
    std::vector<double> outputs_;
    std::vector<double> raw_inputs_;
    std::vector<double> deltas_;

    friend class NeuralNetwork;
};

/**
 * @brief Feedforward neural network with backpropagation training.
 *
 * Supports arbitrary depth, configurable activations per layer,
 * and mini-batch SGD. Thread-safe via shared_mutex:
 *   - Forward() acquires a shared (read) lock
 *   - Train() acquires an exclusive (write) lock
 */
class NeuralNetwork {
public:
    explicit NeuralNetwork(size_t input_size, double learning_rate = 0.1);

    void AddLayer(size_t neurons, std::unique_ptr<ActivationFunction> activation);
    void AddSigmoidLayer(size_t neurons);
    void AddReLULayer(size_t neurons);
    void AddTanhLayer(size_t neurons);

    void SetLearningRate(double rate);
    double GetLearningRate() const;

    std::vector<double> Forward(const std::vector<double>& inputs);
    double Train(const std::vector<double>& inputs, const std::vector<double>& expected);
    std::vector<double> TrainBatch(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& expected,
        size_t epochs = 1000,
        size_t batch_size = 32);

    void PrintArchitecture() const;
    size_t GetLayerCount() const;

private:
    size_t input_size_;
    double learning_rate_;
    std::vector<Layer> layers_;
    mutable std::shared_mutex mutex_;
};

} // namespace Ml
