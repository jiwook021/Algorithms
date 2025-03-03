/**
 * @file NeuralNetwork.cpp
 * @brief Implementation of feedforward neural network with backpropagation.
 */
#include "NeuralNetwork.hpp"

namespace Ml {

// ======================== Layer ========================

Layer::Layer(size_t input_size, size_t output_size,
             std::unique_ptr<ActivationFunction> activation)
    : input_size_(input_size),
      output_size_(output_size),
      activation_func_(std::move(activation)),
      weights_(output_size, std::vector<double>(input_size)),
      biases_(output_size),
      outputs_(output_size),
      raw_inputs_(output_size),
      deltas_(output_size) {

    if (input_size == 0 || output_size == 0) {
        throw std::invalid_argument("Layer sizes must be greater than zero");
    }

    // Xavier / Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double limit = std::sqrt(6.0 / (input_size + output_size));
    std::uniform_real_distribution<double> dist(-limit, limit);

    for (auto& neuron_weights : weights_) {
        for (auto& w : neuron_weights) {
            w = dist(gen);
        }
    }
    for (auto& b : biases_) {
        b = dist(gen);
    }
}

std::vector<double> Layer::Forward(const std::vector<double>& inputs) {
    if (inputs.size() != input_size_) {
        std::stringstream ss;
        ss << "Input size mismatch: expected " << input_size_
           << ", got " << inputs.size();
        throw std::invalid_argument(ss.str());
    }

    for (size_t i = 0; i < output_size_; ++i) {
        double sum = biases_[i];
        for (size_t j = 0; j < input_size_; ++j) {
            sum += weights_[i][j] * inputs[j];
        }
        raw_inputs_[i] = sum;
        outputs_[i] = activation_func_->Activate(sum);
    }
    return outputs_;
}

std::vector<double> Layer::BackwardOutput(const std::vector<double>& expected) {
    if (expected.size() != output_size_) {
        throw std::invalid_argument("Expected output size mismatch");
    }
    for (size_t i = 0; i < output_size_; ++i) {
        double error = outputs_[i] - expected[i];
        deltas_[i] = error * activation_func_->Derivative(raw_inputs_[i]);
    }
    return deltas_;
}

std::vector<double> Layer::BackwardHidden(const Layer& next_layer) {
    for (size_t i = 0; i < output_size_; ++i) {
        double error = 0.0;
        for (size_t j = 0; j < next_layer.output_size_; ++j) {
            error += next_layer.deltas_[j] * next_layer.weights_[j][i];
        }
        deltas_[i] = error * activation_func_->Derivative(raw_inputs_[i]);
    }
    return deltas_;
}

void Layer::UpdateWeights(const std::vector<double>& inputs, double learning_rate) {
    for (size_t i = 0; i < output_size_; ++i) {
        for (size_t j = 0; j < input_size_; ++j) {
            weights_[i][j] -= learning_rate * deltas_[i] * inputs[j];
        }
        biases_[i] -= learning_rate * deltas_[i];
    }
}

// ======================== NeuralNetwork ========================

NeuralNetwork::NeuralNetwork(size_t input_size, double learning_rate)
    : input_size_(input_size), learning_rate_(learning_rate) {
    if (input_size == 0) {
        throw std::invalid_argument("Input size must be greater than zero");
    }
    if (learning_rate <= 0.0 || learning_rate > 1.0) {
        throw std::invalid_argument("Learning rate must be in (0, 1]");
    }
}

void NeuralNetwork::AddLayer(size_t neurons,
                              std::unique_ptr<ActivationFunction> activation) {
    if (neurons == 0) {
        throw std::invalid_argument("Number of neurons must be > 0");
    }
    size_t prev_size = layers_.empty() ? input_size_ : layers_.back().GetOutputSize();
    layers_.emplace_back(prev_size, neurons, std::move(activation));
}

void NeuralNetwork::AddSigmoidLayer(size_t n) {
    AddLayer(n, std::make_unique<Sigmoid>());
}
void NeuralNetwork::AddReLULayer(size_t n) {
    AddLayer(n, std::make_unique<ReLU>());
}
void NeuralNetwork::AddTanhLayer(size_t n) {
    AddLayer(n, std::make_unique<Tanh>());
}

void NeuralNetwork::SetLearningRate(double rate) {
    std::unique_lock lock(mutex_);
    if (rate <= 0.0 || rate > 1.0) {
        throw std::invalid_argument("Learning rate must be in (0, 1]");
    }
    learning_rate_ = rate;
}

double NeuralNetwork::GetLearningRate() const {
    std::shared_lock lock(mutex_);
    return learning_rate_;
}

std::vector<double> NeuralNetwork::Forward(const std::vector<double>& inputs) {
    std::shared_lock lock(mutex_);
    if (layers_.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    if (inputs.size() != input_size_) {
        throw std::invalid_argument("Input size mismatch");
    }
    std::vector<double> current = inputs;
    for (auto& layer : layers_) {
        current = layer.Forward(current);
    }
    return current;
}

double NeuralNetwork::Train(const std::vector<double>& inputs,
                             const std::vector<double>& expected) {
    std::unique_lock lock(mutex_);
    if (layers_.empty()) {
        throw std::runtime_error("Network has no layers");
    }

    // Forward pass -- store each layer's input for weight updates
    std::vector<std::vector<double>> layer_inputs{inputs};
    std::vector<double> current = inputs;
    for (auto& layer : layers_) {
        current = layer.Forward(current);
        layer_inputs.push_back(current);
    }

    // MSE computation
    double error = 0.0;
    const auto& outputs = layers_.back().GetOutputs();
    for (size_t i = 0; i < expected.size(); ++i) {
        double diff = outputs[i] - expected[i];
        error += diff * diff;
    }
    error /= expected.size();

    // Backward pass
    layers_.back().BackwardOutput(expected);
    for (int i = static_cast<int>(layers_.size()) - 2; i >= 0; --i) {
        layers_[static_cast<size_t>(i)].BackwardHidden(
            layers_[static_cast<size_t>(i) + 1]);
    }

    // Weight update
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].UpdateWeights(layer_inputs[i], learning_rate_);
    }

    return error;
}

std::vector<double> NeuralNetwork::TrainBatch(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& expected,
    size_t epochs, size_t batch_size) {

    if (inputs.size() != expected.size() || inputs.empty()) {
        throw std::invalid_argument(
            "Input and expected sizes must match and be non-empty");
    }
    if (epochs == 0) {
        throw std::invalid_argument("Epochs must be > 0");
    }
    if (batch_size == 0 || batch_size > inputs.size()) {
        batch_size = inputs.size();
    }

    std::vector<double> error_history;
    error_history.reserve(epochs);

    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);
        double epoch_error = 0.0;

        for (size_t batch_start = 0; batch_start < inputs.size();
             batch_start += batch_size) {
            size_t cur_batch = std::min(batch_size,
                                        inputs.size() - batch_start);

            for (size_t i = 0; i < cur_batch; ++i) {
                size_t idx = indices[batch_start + i];
                epoch_error += Train(inputs[idx], expected[idx]);
            }
        }

        epoch_error /= inputs.size();
        error_history.push_back(epoch_error);
    }
    return error_history;
}

void NeuralNetwork::PrintArchitecture() const {
    std::shared_lock lock(mutex_);
    std::cout << "Neural Network Architecture:\n";
    std::cout << "Input size: " << input_size_ << "\n";
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& l = layers_[i];
        std::cout << "Layer " << (i + 1) << ": "
                  << l.GetInputSize() << " -> " << l.GetOutputSize()
                  << ", Activation: " << l.GetActivationName() << "\n";
    }
    std::cout << "Learning rate: " << learning_rate_ << "\n";
}

size_t NeuralNetwork::GetLayerCount() const {
    std::shared_lock lock(mutex_);
    return layers_.size();
}

} // namespace Ml
