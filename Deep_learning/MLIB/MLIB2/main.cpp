#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>

//-----------------------------------------------------
// Minimal tensor representation.
// For our fully connected layer, we assume a 1D tensor
// (i.e. a flattened vector).
//-----------------------------------------------------
struct tensor
{
    std::vector<float> data;
    std::vector<size_t> shape;  // e.g., {n} for a 1D tensor.

    // Compute total number of elements.
    size_t size() const {
        size_t s = 1;
        for (auto d : shape)
            s *= d;
        return s;
    }
};

//-----------------------------------------------------
// dlib::fc (Fully Connected Layer)
// Maps an input vector of length input_size to an output vector of length num_outputs.
// Forward: output = W * input + b
// Backward: given gradient on output (dL/dz), compute gradients for weights and biases and update them.
//-----------------------------------------------------
class fc_
{
public:
    size_t input_size;
    size_t num_outputs;
    std::vector<float> weights; // shape: [num_outputs, input_size]
    std::vector<float> bias;    // shape: [num_outputs]

    // Constructor: initializes weights to a small constant and biases to zero.
    fc_(size_t input_size_, size_t num_outputs_)
        : input_size(input_size_), num_outputs(num_outputs_)
    {
        weights.resize(num_outputs * input_size, 0.01f);
        bias.resize(num_outputs, 0.0f);
    }

    // Forward pass: expects the input tensor to be 1D of length input_size.
    tensor forward(const tensor& input)
    {
        assert(input.size() == input_size && "Input size must match fc layer input size");
        tensor output;
        output.shape = { num_outputs };
        output.data.resize(num_outputs, 0.0f);
        // Compute output = W * input + bias.
        for (size_t i = 0; i < num_outputs; ++i)
        {
            float sum = bias[i];
            for (size_t j = 0; j < input_size; ++j)
            {
                sum += weights[i * input_size + j] * input.data[j];
            }
            output.data[i] = sum;
        }
        return output;
    }

    // Backward pass:
    // Given:
    //   - input: the original input tensor (shape {input_size})
    //   - grad_output: gradient of the loss with respect to the output logits (shape {num_outputs})
    //   - learning_rate: scalar learning rate.
    // Updates weights and biases using SGD.
    // Returns (optionally) the gradient with respect to the input.
    std::vector<float> backward(const tensor& input, const std::vector<float>& grad_output, float learning_rate)
    {
        // Compute gradient with respect to input (not used in this one-layer network).
        std::vector<float> grad_input(input_size, 0.0f);
        for (size_t j = 0; j < input_size; j++){
            float sum = 0.0f;
            for (size_t i = 0; i < num_outputs; i++){
                sum += weights[i * input_size + j] * grad_output[i];
            }
            grad_input[j] = sum;
        }
        // Update weights and biases.
        for (size_t i = 0; i < num_outputs; i++){
            for (size_t j = 0; j < input_size; j++){
                float grad_weight = grad_output[i] * input.data[j];
                weights[i * input_size + j] -= learning_rate * grad_weight;
            }
            bias[i] -= learning_rate * grad_output[i];
        }
        return grad_input;
    }
};

//-----------------------------------------------------
// dlib::loss_multiclass_log (Multiclass Classification Loss)
// Computes softmax probabilities and cross-entropy loss.
// Also provides a helper function to compute the gradient of the loss with respect to logits.
//-----------------------------------------------------
class loss_multiclass_log {
public:
    // Forward pass over a batch of samples:
    //   - logits: vector of samples, each sample is a vector of raw scores (one per class)
    //   - labels: ground truth class indices for each sample.
    // Returns: a pair (average_loss, predictions)
    std::pair<float, std::vector<size_t>> forward(
        const std::vector<std::vector<float>>& logits,
        const std::vector<size_t>& labels)
    {
        assert(logits.size() == labels.size());
        size_t num_samples = logits.size();
        float total_loss = 0.0f;
        std::vector<size_t> predictions(num_samples, 0);
        
        for (size_t i = 0; i < num_samples; ++i)
        {
            const auto& logit = logits[i];
            size_t num_classes = logit.size();
            // Compute softmax probabilities in a numerically stable way.
            float max_logit = *std::max_element(logit.begin(), logit.end());
            std::vector<float> exp_logits(num_classes, 0.0f);
            float sum_exp = 0.0f;
            for (size_t j = 0; j < num_classes; ++j)
            {
                exp_logits[j] = std::exp(logit[j] - max_logit);
                sum_exp += exp_logits[j];
            }
            std::vector<float> probs(num_classes, 0.0f);
            for (size_t j = 0; j < num_classes; ++j)
                probs[j] = exp_logits[j] / sum_exp;
            
            // Cross-entropy loss: -log(probability of the true class)
            float sample_loss = -std::log(probs[labels[i]] + 1e-8f);
            total_loss += sample_loss;
            // Prediction is the argmax of logits.
            size_t pred = std::distance(logit.begin(),
                                        std::max_element(logit.begin(), logit.end()));
            predictions[i] = pred;
        }
        float average_loss = total_loss / num_samples;
        return { average_loss, predictions };
    }

    // Helper: For a single sample, compute the gradient with respect to logits.
    // Given:
    //   - logits: raw scores (vector of length num_classes)
    //   - label: ground truth class index
    // Returns: vector of gradients dL/dz (where L = cross-entropy loss).
    std::vector<float> gradient(const std::vector<float>& logits, size_t label)
    {
        size_t num_classes = logits.size();
        float max_logit = *std::max_element(logits.begin(), logits.end());
        std::vector<float> exp_logits(num_classes, 0.0f);
        float sum_exp = 0.0f;
        for (size_t j = 0; j < num_classes; ++j)
        {
            exp_logits[j] = std::exp(logits[j] - max_logit);
            sum_exp += exp_logits[j];
        }
        std::vector<float> grad(num_classes, 0.0f);
        for (size_t j = 0; j < num_classes; ++j)
        {
            float prob = exp_logits[j] / sum_exp;
            // dL/dz = prob - 1{j == label}
            grad[j] = prob - ((j == label) ? 1.0f : 0.0f);
        }
        return grad;
    }
};

//-----------------------------------------------------
// Training and Inference Cycle with Backpropagation
//-----------------------------------------------------
int main()
{
    // --- Dummy Dataset ---
    // Create 5 training samples.
    // Each sample is a 4-dimensional vector.
    // Labels are in {0, 1, 2} (i.e. 3 classes).
    size_t input_size = 4;
    size_t num_classes = 3;
    std::vector<tensor> train_inputs;
    std::vector<size_t> train_labels;
    for (int i = 0; i < 5; i++)
    {
        tensor sample;
        sample.shape = { input_size };
        sample.data.resize(input_size);
        // Create a simple pattern: sample data = [i, i+1, i+2, i+3]
        for (size_t j = 0; j < input_size; j++)
        {
            sample.data[j] = static_cast<float>(i + j);
        }
        train_inputs.push_back(sample);
        // Assign label = i mod num_classes.
        train_labels.push_back(i % num_classes);
    }

    // --- Network Setup ---
    // Use a single fully connected layer that maps 4 inputs to 3 outputs (logits).
    fc_ fc_layer(input_size, num_classes);
    loss_multiclass_log loss_layer;

    // Training hyperparameters.
    float learning_rate = 0.01f;
    int epochs = 50;

    // --- Training Cycle ---
    std::cout << "Training Cycle with Backpropagation:\n";
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float epoch_loss = 0.0f;
        // Process each sample (stochastic gradient descent).
        for (size_t i = 0; i < train_inputs.size(); i++)
        {
            // Forward pass.
            tensor out = fc_layer.forward(train_inputs[i]);
            // Compute loss and gradient for this sample.
            std::vector<float> logits = out.data;
            std::vector<float> grad_logits = loss_layer.gradient(logits, train_labels[i]);
            // Compute sample loss (for reporting).
            float max_logit = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (float v : logits)
                sum_exp += std::exp(v - max_logit);
            float prob = std::exp(logits[train_labels[i]] - max_logit) / sum_exp;
            float sample_loss = -std::log(prob + 1e-8f);
            epoch_loss += sample_loss;
            // Backward pass: update fc_layer parameters.
            fc_layer.backward(train_inputs[i], grad_logits, learning_rate);
        }
        epoch_loss /= train_inputs.size();
        if ((epoch + 1) % 10 == 0)
            std::cout << "Epoch " << epoch + 1 << " - Average Loss: " << epoch_loss << std::endl;
    }

    // --- Inference Cycle ---
    std::cout << "\nInference:\n";
    // Create a test sample.
    tensor test_sample;
    test_sample.shape = { input_size };
    test_sample.data = { 9.0f, 3.0f, 8.0f, 7.0f };
    // Forward pass through the trained network.
    tensor test_out = fc_layer.forward(test_sample);
    // Determine predicted class as the argmax of the output logits.
    size_t predicted = std::distance(test_out.data.begin(),
                                     std::max_element(test_out.data.begin(), test_out.data.end()));
    std::cout << "Test sample input: ";
    for (auto v : test_sample.data)
        std::cout << v << " ";
    std::cout << "\nPredicted class: " << predicted << std::endl;

    // --- Explanation of Inference ---
    //
    // In this example, inference means performing a forward pass
    // through the trained network. Our network consists of a single fully connected layer.
    // The output logits (raw scores) are computed and then the class with the highest score
    // (i.e. argmax) is selected as the predicted label.
    //
    // This inference procedure is suitable for multiclass classification tasks,
    // where the goal is to assign each input sample to one of several discrete classes.
    // In real-world applications, such a network (possibly much deeper and with additional
    // layers/activations) would be used to classify images, text, or other data into categories.
    //
    // The backpropagation and weight update steps here use a simple gradient descent on the
    // softmax cross-entropy loss. In practice, one would use more advanced optimization methods,
    // mini-batching, and regularization techniques.
    
    return 0;
}
