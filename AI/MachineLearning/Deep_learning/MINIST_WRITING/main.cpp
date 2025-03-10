#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <cstdint>
#include <iomanip>
#include <filesystem>
#include <cstring>
#include <cerrno>
#include <numeric>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <omp.h>

// Optional AVX support
#ifdef __AVX2__
#include <immintrin.h>
#endif

// For image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Structure for MNIST dataset
struct MNISTDataset {
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
};

// Forward declaration of helper functions
uint32_t swapEndian(uint32_t value);
MNISTDataset readMNISTDataset(const std::string& imagesFile, const std::string& labelsFile);
std::pair<MNISTDataset, MNISTDataset> createTrainValSplit(const MNISTDataset& trainData, float valRatio = 0.1f);
std::pair<float, float> normalizeDataset(std::vector<std::vector<float>>& images);
void normalizeImage(std::vector<float>& image, float mean, float stdDev);

/**
 * Optimized Neural Network with batch processing support
 * 
 * This enhanced MLP uses:
 * - Parallelized batch processing with OpenMP
 * - SIMD vectorization where available
 * - Cache-friendly memory access patterns
 * - Optimized matrix operations
 * - Adam optimizer with batch updates
 */
class OptimizedNeuralNetwork {
private:
    // Network architecture
    int inputSize;
    int hiddenSize;
    int outputSize;
    
    // Network parameters
    std::vector<float> inputToHiddenWeights;
    std::vector<float> hiddenBiases;
    std::vector<float> hiddenToOutputWeights;
    std::vector<float> outputBiases;
    
    // Adam optimizer parameters
    std::vector<float> m_ih, v_ih; // Momentum and velocity for input-hidden weights
    std::vector<float> m_hb, v_hb; // Momentum and velocity for hidden biases
    std::vector<float> m_ho, v_ho; // Momentum and velocity for hidden-output weights
    std::vector<float> m_ob, v_ob; // Momentum and velocity for output biases
    float beta1 = 0.9;   // Exponential decay rate for first moment
    float beta2 = 0.999; // Exponential decay rate for second moment
    float epsilon = 1e-8; // Small constant for numerical stability
    int t = 0;          // Time step (updated during training)
    
    // Thread-safe random number generation
    std::mutex rngMutex;
    std::mt19937 rng;
    
    // ReLU activation function and its derivative
    static float relu(float x) {
        return std::max(0.0f, x);
    }
    
    static float reluDerivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }
    
    // Vectorized ReLU implementation
    void vectorizedRelu(float* data, int size) const {
#ifdef __AVX2__
        // Process 8 elements at a time using AVX2
        int i = 0;
        for (; i <= size - 8; i += 8) {
            __m256 values = _mm256_loadu_ps(data + i);
            __m256 zeros = _mm256_setzero_ps();
            __m256 result = _mm256_max_ps(values, zeros);
            _mm256_storeu_ps(data + i, result);
        }
        
        // Process remaining elements
        for (; i < size; i++) {
            data[i] = std::max(0.0f, data[i]);
        }
#else
        // Fallback for non-AVX2 systems
        for (int i = 0; i < size; i++) {
            data[i] = std::max(0.0f, data[i]);
        }
#endif
    }
    
    // Vectorized ReLU derivative
    void vectorizedReluDerivative(const float* input, float* output, int size) const {
#ifdef __AVX2__
        int i = 0;
        for (; i <= size - 8; i += 8) {
            __m256 values = _mm256_loadu_ps(input + i);
            __m256 zeros = _mm256_setzero_ps();
            __m256 ones = _mm256_set1_ps(1.0f);
            __m256 mask = _mm256_cmp_ps(values, zeros, _CMP_GT_OQ);
            __m256 result = _mm256_and_ps(ones, mask);
            _mm256_storeu_ps(output + i, result);
        }
        
        for (; i < size; i++) {
            output[i] = input[i] > 0.0f ? 1.0f : 0.0f;
        }
#else
        for (int i = 0; i < size; i++) {
            output[i] = input[i] > 0.0f ? 1.0f : 0.0f;
        }
#endif
    }
    
    // Softmax activation function for a single sample
    std::vector<float> softmax(const std::vector<float>& x) const {
        std::vector<float> output(x.size());
        float maxVal = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < x.size(); i++) {
            output[i] = std::exp(x[i] - maxVal); // Subtract max for numerical stability
            sum += output[i];
        }
        
        // Normalize
        for (size_t i = 0; i < x.size(); i++) {
            output[i] /= sum;
        }
        
        return output;
    }
    
    // Optimized softmax for a batch of samples
    void batchSoftmax(const std::vector<std::vector<float>>& inputs, 
                     std::vector<std::vector<float>>& outputs) const {
        const int batchSize = inputs.size();
        
        // Process each sample in parallel
        #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            const std::vector<float>& input = inputs[b];
            std::vector<float>& output = outputs[b];
            
            // Find max value for numerical stability
            float maxVal = *std::max_element(input.begin(), input.end());
            
            // Compute exp(x - max) and sum
            float sum = 0.0f;
            for (size_t i = 0; i < input.size(); i++) {
                output[i] = std::exp(input[i] - maxVal);
                sum += output[i];
            }
            
            // Normalize
            float invSum = 1.0f / sum;
            for (size_t i = 0; i < output.size(); i++) {
                output[i] *= invSum;
            }
        }
    }
    
public:
    OptimizedNeuralNetwork(int inputSize, int hiddenSize = 128, int outputSize = 10) 
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), rng(std::random_device{}()) {
        
        // Initialize weights
        inputToHiddenWeights.resize(inputSize * hiddenSize);
        hiddenBiases.resize(hiddenSize);
        hiddenToOutputWeights.resize(hiddenSize * outputSize);
        outputBiases.resize(outputSize);
        
        // Initialize Adam optimizer parameters
        m_ih.resize(inputSize * hiddenSize, 0.0f);
        v_ih.resize(inputSize * hiddenSize, 0.0f);
        m_hb.resize(hiddenSize, 0.0f);
        v_hb.resize(hiddenSize, 0.0f);
        m_ho.resize(hiddenSize * outputSize, 0.0f);
        v_ho.resize(hiddenSize * outputSize, 0.0f);
        m_ob.resize(outputSize, 0.0f);
        v_ob.resize(outputSize, 0.0f);
        
        // Initialize weights with appropriate scaling
        initializeWeights();
    }
    
    // Initialize weights with appropriate scaling
    void initializeWeights() {
        std::lock_guard<std::mutex> lock(rngMutex); // Thread-safe RNG access
        
        // He initialization for hidden layer
        float scale_ih = std::sqrt(2.0f / inputSize);
        std::normal_distribution<float> dist_ih(0, scale_ih);
        
        // Xavier initialization for output layer
        float scale_ho = std::sqrt(2.0f / (hiddenSize + outputSize));
        std::normal_distribution<float> dist_ho(0, scale_ho);
        
        // Random initialization of weights
        for (int i = 0; i < inputSize * hiddenSize; i++) {
            inputToHiddenWeights[i] = dist_ih(rng);
        }
        
        std::fill(hiddenBiases.begin(), hiddenBiases.end(), 0.0f);
        
        for (int i = 0; i < hiddenSize * outputSize; i++) {
            hiddenToOutputWeights[i] = dist_ho(rng);
        }
        
        std::fill(outputBiases.begin(), outputBiases.end(), 0.0f);
    }
    
    // Forward pass for a single sample
    std::vector<float> forward(const std::vector<float>& input) const {
        // Hidden layer activations
        std::vector<float> hiddenLayerInput(hiddenSize, 0.0f);
        std::vector<float> hiddenLayerOutput(hiddenSize);
        
        // Initialize with biases
        std::copy(hiddenBiases.begin(), hiddenBiases.end(), hiddenLayerInput.begin());
        
        // Input to hidden layer
        for (int i = 0; i < inputSize; i++) {
            const float inputVal = input[i];
            for (int j = 0; j < hiddenSize; j++) {
                hiddenLayerInput[j] += inputVal * inputToHiddenWeights[i * hiddenSize + j];
            }
        }
        
        // Apply ReLU activation
        for (int j = 0; j < hiddenSize; j++) {
            hiddenLayerOutput[j] = relu(hiddenLayerInput[j]);
        }
        
        // Output layer activations
        std::vector<float> outputLayerInput(outputSize, 0.0f);
        
        // Initialize with biases
        std::copy(outputBiases.begin(), outputBiases.end(), outputLayerInput.begin());
        
        // Hidden to output layer
        for (int j = 0; j < hiddenSize; j++) {
            const float hiddenVal = hiddenLayerOutput[j];
            for (int k = 0; k < outputSize; k++) {
                outputLayerInput[k] += hiddenVal * hiddenToOutputWeights[j * outputSize + k];
            }
        }
        
        // Apply softmax to get probabilities
        return softmax(outputLayerInput);
    }
    
    // Forward pass for a batch of samples
    std::vector<std::vector<float>> batchForward(const std::vector<std::vector<float>>& inputs) const {
        const int batchSize = inputs.size();
        
        // Prepare output containers
        std::vector<std::vector<float>> hiddenLayerInputs(batchSize, std::vector<float>(hiddenSize));
        std::vector<std::vector<float>> hiddenLayerOutputs(batchSize, std::vector<float>(hiddenSize));
        std::vector<std::vector<float>> outputLayerInputs(batchSize, std::vector<float>(outputSize));
        std::vector<std::vector<float>> outputLayerOutputs(batchSize, std::vector<float>(outputSize));
        
        // Compute hidden layer activations in parallel
        #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            const auto& input = inputs[b];
            auto& hiddenInput = hiddenLayerInputs[b];
            auto& hiddenOutput = hiddenLayerOutputs[b];
            
            // Initialize with biases
            std::copy(hiddenBiases.begin(), hiddenBiases.end(), hiddenInput.begin());
            
            // Matrix multiplication: hidden = input * weights + biases
            for (int i = 0; i < inputSize; i++) {
                const float inputVal = input[i];
                for (int j = 0; j < hiddenSize; j++) {
                    hiddenInput[j] += inputVal * inputToHiddenWeights[i * hiddenSize + j];
                }
            }
            
            // Apply ReLU activation
            for (int j = 0; j < hiddenSize; j++) {
                hiddenOutput[j] = relu(hiddenInput[j]);
            }
        }
        
        // Compute output layer activations in parallel
        #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            const auto& hiddenOutput = hiddenLayerOutputs[b];
            auto& outputInput = outputLayerInputs[b];
            
            // Initialize with biases
            std::copy(outputBiases.begin(), outputBiases.end(), outputInput.begin());
            
            // Matrix multiplication: output = hidden * weights + biases
            for (int j = 0; j < hiddenSize; j++) {
                const float hiddenVal = hiddenOutput[j];
                for (int k = 0; k < outputSize; k++) {
                    outputInput[k] += hiddenVal * hiddenToOutputWeights[j * outputSize + k];
                }
            }
        }
        
        // Apply softmax to get probabilities
        batchSoftmax(outputLayerInputs, outputLayerOutputs);
        
        return outputLayerOutputs;
    }
    
    // Returns the predicted digit (highest probability)
    int predict(const std::vector<float>& input) const {
        std::vector<float> probs = forward(input);
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }
    
    // Batch training using mini-batch gradient descent and Adam optimizer
    void batchTrain(const std::vector<std::vector<float>>& inputs, 
                  const std::vector<std::vector<float>>& targets, 
                  float learningRate) {
        t++; // Increment time step
        const int batchSize = inputs.size();
        
        // Intermediate activations and gradients
        std::vector<std::vector<float>> hiddenLayerInputs(batchSize, std::vector<float>(hiddenSize));
        std::vector<std::vector<float>> hiddenLayerOutputs(batchSize, std::vector<float>(hiddenSize));
        std::vector<std::vector<float>> outputLayerInputs(batchSize, std::vector<float>(outputSize));
        std::vector<std::vector<float>> outputLayerOutputs(batchSize, std::vector<float>(outputSize));
        std::vector<std::vector<float>> outputGradients(batchSize, std::vector<float>(outputSize));
        std::vector<std::vector<float>> hiddenGradients(batchSize, std::vector<float>(hiddenSize));
        
        // Forward pass - calculate and store all intermediate activations
        // ================================================================
        
        // Compute hidden layer activations in parallel
        #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            const auto& input = inputs[b];
            auto& hiddenInput = hiddenLayerInputs[b];
            auto& hiddenOutput = hiddenLayerOutputs[b];
            
            // Initialize with biases
            std::copy(hiddenBiases.begin(), hiddenBiases.end(), hiddenInput.begin());
            
            // Matrix multiplication: hidden = input * weights + biases
            for (int i = 0; i < inputSize; i++) {
                const float inputVal = input[i];
                for (int j = 0; j < hiddenSize; j++) {
                    hiddenInput[j] += inputVal * inputToHiddenWeights[i * hiddenSize + j];
                }
            }
            
            // Apply ReLU activation
            for (int j = 0; j < hiddenSize; j++) {
                hiddenOutput[j] = relu(hiddenInput[j]);
            }
        }
        
        // Compute output layer activations in parallel
        #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            const auto& hiddenOutput = hiddenLayerOutputs[b];
            auto& outputInput = outputLayerInputs[b];
            
            // Initialize with biases
            std::copy(outputBiases.begin(), outputBiases.end(), outputInput.begin());
            
            // Matrix multiplication: output = hidden * weights + biases
            for (int j = 0; j < hiddenSize; j++) {
                const float hiddenVal = hiddenOutput[j];
                for (int k = 0; k < outputSize; k++) {
                    outputInput[k] += hiddenVal * hiddenToOutputWeights[j * outputSize + k];
                }
            }
        }
        
        // Apply softmax to get probabilities
        batchSoftmax(outputLayerInputs, outputLayerOutputs);
        
        // Backward pass - calculate gradients
        // ===================================
        
        // Calculate output layer gradients (softmax with cross-entropy derivative) in parallel
        #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            const auto& output = outputLayerOutputs[b];
            const auto& target = targets[b];
            auto& outGrad = outputGradients[b];
            
            for (int k = 0; k < outputSize; k++) {
                outGrad[k] = output[k] - target[k];
            }
        }
        
        // Calculate hidden layer gradients in parallel
        #pragma omp parallel for
        for (int b = 0; b < batchSize; b++) {
            const auto& outGrad = outputGradients[b];
            const auto& hiddenInput = hiddenLayerInputs[b];
            auto& hiddenGrad = hiddenGradients[b];
            
            std::fill(hiddenGrad.begin(), hiddenGrad.end(), 0.0f);
            
            // Calculate gradients
            for (int j = 0; j < hiddenSize; j++) {
                float sum = 0.0f;
                for (int k = 0; k < outputSize; k++) {
                    sum += outGrad[k] * hiddenToOutputWeights[j * outputSize + k];
                }
                hiddenGrad[j] = sum * (hiddenInput[j] > 0.0f ? 1.0f : 0.0f);
            }
        }
        
        // Accumulate weight gradients
        // ===========================
        
        // Initialize weight gradient accumulators
        std::vector<float> ihGradients(inputSize * hiddenSize, 0.0f);
        std::vector<float> hbGradients(hiddenSize, 0.0f);
        std::vector<float> hoGradients(hiddenSize * outputSize, 0.0f);
        std::vector<float> obGradients(outputSize, 0.0f);
        
        // Accumulate gradients across the batch
        for (int b = 0; b < batchSize; b++) {
            // Accumulate hidden-to-output weights gradients
            const auto& hiddenOutput = hiddenLayerOutputs[b];
            const auto& outGrad = outputGradients[b];
            
            for (int j = 0; j < hiddenSize; j++) {
                const float hiddenVal = hiddenOutput[j];
                for (int k = 0; k < outputSize; k++) {
                    hoGradients[j * outputSize + k] += outGrad[k] * hiddenVal;
                }
            }
            
            // Accumulate output bias gradients
            for (int k = 0; k < outputSize; k++) {
                obGradients[k] += outGrad[k];
            }
            
            // Accumulate input-to-hidden weights gradients
            const auto& input = inputs[b];
            const auto& hiddenGrad = hiddenGradients[b];
            
            for (int i = 0; i < inputSize; i++) {
                const float inputVal = input[i];
                for (int j = 0; j < hiddenSize; j++) {
                    ihGradients[i * hiddenSize + j] += hiddenGrad[j] * inputVal;
                }
            }
            
            // Accumulate hidden bias gradients
            for (int j = 0; j < hiddenSize; j++) {
                hbGradients[j] += hiddenGrad[j];
            }
        }
        
        // Normalize gradients by batch size
        const float scaleFactor = 1.0f / batchSize;
        
        // Scale the gradients
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                for (auto& grad : ihGradients) grad *= scaleFactor;
            }
            #pragma omp section
            {
                for (auto& grad : hbGradients) grad *= scaleFactor;
            }
            #pragma omp section
            {
                for (auto& grad : hoGradients) grad *= scaleFactor;
            }
            #pragma omp section
            {
                for (auto& grad : obGradients) grad *= scaleFactor;
            }
        }
        
        // Apply Adam optimizer to update weights
        // =====================================
        
        // Compute bias correction terms
        float correction1 = 1.0f - std::pow(beta1, t);
        float correction2 = 1.0f - std::pow(beta2, t);
        float correctedLR = learningRate * std::sqrt(correction2) / correction1;
        
        // Update input-to-hidden weights
        #pragma omp parallel for
        for (int i = 0; i < inputSize * hiddenSize; i++) {
            // Update biased first moment estimate
            m_ih[i] = beta1 * m_ih[i] + (1 - beta1) * ihGradients[i];
            // Update biased second moment estimate
            v_ih[i] = beta2 * v_ih[i] + (1 - beta2) * ihGradients[i] * ihGradients[i];
            
            // Update parameters with corrected learning rate
            inputToHiddenWeights[i] -= correctedLR * m_ih[i] / (std::sqrt(v_ih[i]) + epsilon);
        }
        
        // Update hidden biases
        #pragma omp parallel for
        for (int j = 0; j < hiddenSize; j++) {
            m_hb[j] = beta1 * m_hb[j] + (1 - beta1) * hbGradients[j];
            v_hb[j] = beta2 * v_hb[j] + (1 - beta2) * hbGradients[j] * hbGradients[j];
            
            hiddenBiases[j] -= correctedLR * m_hb[j] / (std::sqrt(v_hb[j]) + epsilon);
        }
        
        // Update hidden-to-output weights
        #pragma omp parallel for
        for (int i = 0; i < hiddenSize * outputSize; i++) {
            m_ho[i] = beta1 * m_ho[i] + (1 - beta1) * hoGradients[i];
            v_ho[i] = beta2 * v_ho[i] + (1 - beta2) * hoGradients[i] * hoGradients[i];
            
            hiddenToOutputWeights[i] -= correctedLR * m_ho[i] / (std::sqrt(v_ho[i]) + epsilon);
        }
        
        // Update output biases
        #pragma omp parallel for
        for (int k = 0; k < outputSize; k++) {
            m_ob[k] = beta1 * m_ob[k] + (1 - beta1) * obGradients[k];
            v_ob[k] = beta2 * v_ob[k] + (1 - beta2) * obGradients[k] * obGradients[k];
            
            outputBiases[k] -= correctedLR * m_ob[k] / (std::sqrt(v_ob[k]) + epsilon);
        }
    }
    
    // Save model to file
    bool saveModel(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return false;
        }
        
        // Write architecture
        file.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
        file.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(hiddenSize));
        file.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));
        
        // Write weights and biases
        file.write(reinterpret_cast<const char*>(inputToHiddenWeights.data()), inputToHiddenWeights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(hiddenBiases.data()), hiddenBiases.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(hiddenToOutputWeights.data()), hiddenToOutputWeights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(outputBiases.data()), outputBiases.size() * sizeof(float));
        
        return true;
    }
    
    // Load model from file
    bool loadModel(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening model file: " << filename << std::endl;
            return false;
        }
        
        // Read architecture
        int storedInputSize, storedHiddenSize, storedOutputSize;
        file.read(reinterpret_cast<char*>(&storedInputSize), sizeof(storedInputSize));
        file.read(reinterpret_cast<char*>(&storedHiddenSize), sizeof(storedHiddenSize));
        file.read(reinterpret_cast<char*>(&storedOutputSize), sizeof(storedOutputSize));
        
        // Check if architecture matches
        if (storedInputSize != inputSize || storedHiddenSize != hiddenSize || storedOutputSize != outputSize) {
            std::cerr << "Model architecture mismatch." << std::endl;
            return false;
        }
        
        // Read weights and biases
        file.read(reinterpret_cast<char*>(inputToHiddenWeights.data()), inputToHiddenWeights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(hiddenBiases.data()), hiddenBiases.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(hiddenToOutputWeights.data()), hiddenToOutputWeights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(outputBiases.data()), outputBiases.size() * sizeof(float));
        
        return true;
    }
};

/**
 * Optimized batch data augmentation
 * - Applies transformations in parallel across the batch
 * - Uses simplified elastic distortion for speed
 * - Caches displacement fields for better performance
 */
class BatchAugmenter {
private:
    std::mutex rngMutex;
    std::mt19937 rng;
    float intensity;
    
    // Apply Gaussian noise to the image
    void applyNoise(std::vector<float>& image, float stddev, std::mt19937& gen) {
        std::normal_distribution<float> noise(0.0f, stddev);
        
        // Apply noise in chunks for better cache locality
        const int chunkSize = 64; 
        
        for (size_t i = 0; i < image.size(); i += chunkSize) {
            size_t end = std::min(i + chunkSize, image.size());
            for (size_t j = i; j < end; j++) {
                image[j] += noise(gen);
                image[j] = std::max(0.0f, std::min(1.0f, image[j]));
            }
        }
    }
    
    // Simplified elastic distortion for faster computation
    void simpleElasticDistortion(std::vector<float>& image, int width, int height, std::mt19937& gen) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        // Use a smaller grid for displacement field (faster)
        const int gridSize = 7;
        std::vector<float> dx(gridSize * gridSize);
        std::vector<float> dy(gridSize * gridSize);
        
        // Generate random displacement fields
        for (int i = 0; i < gridSize * gridSize; i++) {
            dx[i] = dist(gen) * 2.0f * intensity;
            dy[i] = dist(gen) * 2.0f * intensity;
        }
        
        std::vector<float> result(width * height, 0.0f);
        
        // Apply displacement with bilinear interpolation
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Map pixel to grid coordinates
                float gx = (x / static_cast<float>(width)) * (gridSize - 1);
                float gy = (y / static_cast<float>(height)) * (gridSize - 1);
                
                // Grid cell coordinates
                int gx0 = std::max(0, std::min(gridSize - 2, static_cast<int>(gx)));
                int gy0 = std::max(0, std::min(gridSize - 2, static_cast<int>(gy)));
                int gx1 = gx0 + 1;
                int gy1 = gy0 + 1;
                
                // Interpolation weights
                float wx = gx - gx0;
                float wy = gy - gy0;
                
                // Interpolate displacement
                float dispX = (1-wx)*(1-wy) * dx[gy0*gridSize + gx0] +
                              wx*(1-wy) * dx[gy0*gridSize + gx1] +
                              (1-wx)*wy * dx[gy1*gridSize + gx0] +
                              wx*wy * dx[gy1*gridSize + gx1];
                
                float dispY = (1-wx)*(1-wy) * dy[gy0*gridSize + gx0] +
                              wx*(1-wy) * dy[gy0*gridSize + gx1] +
                              (1-wx)*wy * dy[gy1*gridSize + gx0] +
                              wx*wy * dy[gy1*gridSize + gx1];
                
                // Apply displacement
                float srcX = x + dispX;
                float srcY = y + dispY;
                
                // Skip if outside the image
                if (srcX < 0 || srcX >= width - 1 || srcY < 0 || srcY >= height - 1) {
                    continue;
                }
                
                // Bilinear interpolation
                int x0 = static_cast<int>(srcX);
                int y0 = static_cast<int>(srcY);
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                float wx_src = srcX - x0;
                float wy_src = srcY - y0;
                
                result[y * width + x] = 
                    (1-wx_src)*(1-wy_src) * image[y0*width + x0] +
                    wx_src*(1-wy_src) * image[y0*width + x1] +
                    (1-wx_src)*wy_src * image[y1*width + x0] +
                    wx_src*wy_src * image[y1*width + x1];
            }
        }
        
        image = result;
    }
    
    // Apply rotation to the image
    void applyRotation(std::vector<float>& image, int width, int height, float angleDegrees, std::mt19937& gen) {
        float angleRad = angleDegrees * M_PI / 180.0f;
        float cosA = std::cos(angleRad);
        float sinA = std::sin(angleRad);
        
        std::vector<float> result(width * height, 0.0f);
        
        // Rotation around the center of the image
        float centerX = width / 2.0f;
        float centerY = height / 2.0f;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Translate coordinates to origin
                float xOrigin = x - centerX;
                float yOrigin = y - centerY;
                
                // Apply rotation
                float xRotated = xOrigin * cosA - yOrigin * sinA;
                float yRotated = xOrigin * sinA + yOrigin * cosA;
                
                // Translate back
                float xSource = xRotated + centerX;
                float ySource = yRotated + centerY;
                
                // Skip if outside the image
                if (xSource < 0 || xSource >= width - 1 || ySource < 0 || ySource >= height - 1) {
                    continue;
                }
                
                // Bilinear interpolation
                int x0 = static_cast<int>(std::floor(xSource));
                int y0 = static_cast<int>(std::floor(ySource));
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                float wx = xSource - x0;
                float wy = ySource - y0;
                
                // Calculate interpolated value
                float val = 0.0f;
                val += image[y0 * width + x0] * (1 - wx) * (1 - wy);
                val += image[y0 * width + x1] * wx * (1 - wy);
                val += image[y1 * width + x0] * (1 - wx) * wy;
                val += image[y1 * width + x1] * wx * wy;
                
                result[y * width + x] = val;
            }
        }
        
        image = result;
    }
    
    // Apply shift to the image
    void applyShift(std::vector<float>& image, int width, int height, int shiftX, int shiftY) {
        std::vector<float> result(width * height, 0.0f);
        
        // Optimize for case of no shift
        if (shiftX == 0 && shiftY == 0) {
            return;
        }
        
        for (int y = 0; y < height; y++) {
            int sourceY = y - shiftY;
            if (sourceY < 0 || sourceY >= height) continue;
            
            for (int x = 0; x < width; x++) {
                int sourceX = x - shiftX;
                if (sourceX < 0 || sourceX >= width) continue;
                
                result[y * width + x] = image[sourceY * width + sourceX];
            }
        }
        
        image = result;
    }
    
public:
    BatchAugmenter(float intensity = 0.5f) 
        : rng(std::random_device{}()), intensity(intensity) {}
    
    // Augment a batch of images in parallel
    std::vector<std::vector<float>> augmentBatch(
        const std::vector<std::vector<float>>& images, 
        int width = 28, int height = 28) {
        
        int batchSize = images.size();
        std::vector<std::vector<float>> augmentedBatch(batchSize);
        
        // Process each image in parallel
        #pragma omp parallel for
        for (int i = 0; i < batchSize; i++) {
            // Thread-local random generator
            std::mt19937 localRng(rng() + i);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            // Start with a copy of the original image
            augmentedBatch[i] = images[i];
            
            // Apply transformations based on intensity
            if (dist(localRng) < intensity) {
                simpleElasticDistortion(augmentedBatch[i], width, height, localRng);
            }
            
            if (dist(localRng) < intensity) {
                // Apply small random rotation
                float angle = (dist(localRng) * 2.0f - 1.0f) * 15.0f * intensity;
                applyRotation(augmentedBatch[i], width, height, angle, localRng);
            }
            
            if (dist(localRng) < intensity) {
                // Apply small random shift
                int shiftX = static_cast<int>((dist(localRng) * 2.0f - 1.0f) * 3.0f * intensity);
                int shiftY = static_cast<int>((dist(localRng) * 2.0f - 1.0f) * 3.0f * intensity);
                applyShift(augmentedBatch[i], width, height, shiftX, shiftY);
            }
            
            // Apply noise with lower probability
            if (dist(localRng) < intensity * 0.3f) {
                applyNoise(augmentedBatch[i], 0.05f * intensity, localRng);
            }
        }
        
        return augmentedBatch;
    }
};

/**
 * Optimized batch creation for training
 * - Creates stratified batches for better class balance
 * - Optimizes memory usage and access patterns
 */
class BatchProcessor {
private:
    std::mutex rngMutex;
    std::mt19937 rng;
    
public:
    BatchProcessor() : rng(std::random_device{}()) {}
    
    // Create stratified mini-batches (balances classes within batches)
    std::pair<std::vector<std::vector<std::vector<float>>>, 
             std::vector<std::vector<std::vector<float>>>>
    createStratifiedMiniBatches(
        const std::vector<std::vector<float>>& images,
        const std::vector<int>& labels,
        int batchSize,
        int numClasses = 10) {
        
        // Group samples by class
        std::vector<std::vector<size_t>> classSampleIndices(numClasses);
        for (size_t i = 0; i < labels.size(); i++) {
            classSampleIndices[labels[i]].push_back(i);
        }
        
        // Shuffle each class's samples
        {
            std::lock_guard<std::mutex> lock(rngMutex);
            for (auto& classIndices : classSampleIndices) {
                std::shuffle(classIndices.begin(), classIndices.end(), rng);
            }
        }
        
        // Calculate number of samples per class
        std::vector<int> samplesPerClass(numClasses);
        int totalSamples = 0;
        
        for (int c = 0; c < numClasses; c++) {
            samplesPerClass[c] = classSampleIndices[c].size();
            totalSamples += samplesPerClass[c];
        }
        
        // Calculate number of batches
        size_t numSamples = images.size();
        size_t numBatches = (numSamples + static_cast<size_t>(batchSize) - 1) / static_cast<size_t>(batchSize);
        
        // Initialize batch storage
        std::vector<std::vector<std::vector<float>>> imageBatches(numBatches);
        std::vector<std::vector<std::vector<float>>> targetBatches(numBatches);
        
        for (size_t i = 0; i < numBatches; i++) {
            imageBatches[i].resize(static_cast<size_t>(batchSize));
            targetBatches[i].resize(static_cast<size_t>(batchSize));
            
            for (int j = 0; j < batchSize; j++) {
                targetBatches[i][j].resize(numClasses, 0.0f);
            }
        }
        
        // Class pointers for tracking position in each class's samples
        std::vector<size_t> classPos(numClasses, 0);
        
        // Fill batches in a stratified manner
        for (size_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            size_t sampleIdx = 0;
            
            // Fill batch with a balanced mix of classes
            while (sampleIdx < static_cast<size_t>(batchSize) && sampleIdx < numSamples) {
                for (int c = 0; c < numClasses && sampleIdx < static_cast<size_t>(batchSize); c++) {
                    if (classPos[c] < classSampleIndices[c].size()) {
                        size_t imageIdx = classSampleIndices[c][classPos[c]];
                        
                        // Copy image data
                        imageBatches[batchIdx][sampleIdx] = images[imageIdx];
                        
                        // Create one-hot encoding
                        std::fill(targetBatches[batchIdx][sampleIdx].begin(), 
                                 targetBatches[batchIdx][sampleIdx].end(), 0.0f);
                        targetBatches[batchIdx][sampleIdx][c] = 1.0f;
                        
                        classPos[c]++;
                        sampleIdx++;
                    }
                }
                
                // If we've used all samples from all classes but the batch isn't full yet,
                // break out of the loop
                bool allClassesExhausted = true;
                for (int c = 0; c < numClasses; c++) {
                    if (classPos[c] < classSampleIndices[c].size()) {
                        allClassesExhausted = false;
                        break;
                    }
                }
                
                if (allClassesExhausted) break;
            }
            
            // If this is the last batch and it's not full, we pad it
            if (sampleIdx < static_cast<size_t>(batchSize)) {
                for (size_t j = sampleIdx; j < static_cast<size_t>(batchSize); j++) {
                    // Duplicate the first sample to maintain batch size
                    imageBatches[batchIdx][j] = imageBatches[batchIdx][0];
                    targetBatches[batchIdx][j] = targetBatches[batchIdx][0];
                }
            }
        }
        
        return {imageBatches, targetBatches};
    }
    
    // Create mini-batches with optimized memory layout (simple version)
    std::pair<std::vector<std::vector<std::vector<float>>>, 
             std::vector<std::vector<int>>>
    createMiniBatches(
        const std::vector<std::vector<float>>& images,
        const std::vector<int>& labels,
        int batchSize) {
        
        size_t numSamples = images.size();
        size_t numBatches = (numSamples + static_cast<size_t>(batchSize) - 1) / static_cast<size_t>(batchSize);
        
        // Create shuffled indices
        std::vector<size_t> indices(numSamples);
        std::iota(indices.begin(), indices.end(), 0);
        
        {
            std::lock_guard<std::mutex> lock(rngMutex);
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        
        // Create batches
        std::vector<std::vector<std::vector<float>>> batchImages(numBatches);
        std::vector<std::vector<int>> batchLabels(numBatches);
        
        for (size_t i = 0; i < numBatches; i++) {
            size_t batchStart = i * static_cast<size_t>(batchSize);
            size_t currentBatchSize = std::min(static_cast<size_t>(batchSize), numSamples - batchStart);
            
            batchImages[i].resize(currentBatchSize);
            batchLabels[i].resize(currentBatchSize);
            
            for (size_t j = 0; j < currentBatchSize; j++) {
                size_t idx = indices[batchStart + j];
                batchImages[i][j] = images[idx];
                batchLabels[i][j] = labels[idx];
            }
        }
        
        return {batchImages, batchLabels};
    }
};

/**
 * Optimized training function that integrates all performance improvements
 */
void trainOptimizedNetwork(
    OptimizedNeuralNetwork& model,
    const std::vector<std::vector<float>>& trainImages,
    const std::vector<int>& trainLabels,
    const std::vector<std::vector<float>>& valImages,
    const std::vector<int>& valLabels,
    int epochs,
    float initialLearningRate,
    int batchSize,
    float augmentationIntensity = 0.5f) {
    
    std::cout << "Training optimized network with " << trainImages.size() << " samples..." << std::endl;
    
    // Initialize tools for batch processing
    BatchProcessor batchProcessor;
    BatchAugmenter augmenter(augmentationIntensity);
    
    float learningRate = initialLearningRate;
    float bestValAccuracy = 0.0f;
    int patienceCounter = 0;
    const int patience = 5; // Early stopping patience
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Pre-compute one-hot encoded validation targets
    std::vector<std::vector<float>> valTargets(valLabels.size(), std::vector<float>(10, 0.0f));
    for (size_t i = 0; i < valLabels.size(); i++) {
        valTargets[i][valLabels[i]] = 1.0f;
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Apply learning rate decay
        if (epoch > 0 && epoch % 5 == 0) {
            learningRate *= 0.7f;
        }
        
        // Create stratified mini-batches for better class balance
        auto [imageBatches, targetBatches] = 
            batchProcessor.createStratifiedMiniBatches(trainImages, trainLabels, batchSize);
        
        // Training metrics
        int trainCorrect = 0;
        float trainLoss = 0.0f;
        
        // Process each mini-batch
        for (size_t batchIdx = 0; batchIdx < imageBatches.size(); ++batchIdx) {
            const auto& batchImages = imageBatches[batchIdx];
            const auto& batchTargets = targetBatches[batchIdx];
            
            // Apply batch augmentation with 50% probability
            std::vector<std::vector<float>> augmentedBatch;
            if (augmentationIntensity > 0.0f && (static_cast<float>(rand()) / RAND_MAX) < 0.5f) {
                augmentedBatch = augmenter.augmentBatch(batchImages);
            } else {
                augmentedBatch = batchImages;
            }
            
            // Train on this batch
            model.batchTrain(augmentedBatch, batchTargets, learningRate);
            
            // Calculate metrics for this batch
            auto outputProbs = model.batchForward(augmentedBatch);
            
            // Calculate accuracy and loss
            for (size_t i = 0; i < static_cast<size_t>(batchSize) && i < trainImages.size() - batchIdx * static_cast<size_t>(batchSize); i++) {
                // Find predicted class
                int predicted = std::max_element(outputProbs[i].begin(), outputProbs[i].end()) - 
                               outputProbs[i].begin();
                int actual = std::max_element(batchTargets[i].begin(), batchTargets[i].end()) -
                            batchTargets[i].begin();
                
                if (predicted == actual) {
                    trainCorrect++;
                }
                
                // Calculate cross-entropy loss
                float sampleLoss = -std::log(std::max(outputProbs[i][actual], 1e-7f));
                trainLoss += sampleLoss;
            }
            
            // Print progress for every 10% of batches
            if (batchIdx % std::max(1UL, imageBatches.size() / 10) == 0) {
                std::cout << "  Batch " << batchIdx << "/" << imageBatches.size() << " processed" << std::endl;
            }
        }
        
        // Calculate training metrics
        float trainAccuracy = static_cast<float>(trainCorrect) / trainImages.size();
        trainLoss /= trainImages.size();
        
        // Evaluate on validation set
        int valCorrect = 0;
        float valLoss = 0.0f;
        
        // Process validation set in batches for memory efficiency
        const int valBatchSize = 100;
        for (size_t i = 0; i < valImages.size(); i += static_cast<size_t>(valBatchSize)) {
            size_t batchEnd = std::min(i + static_cast<size_t>(valBatchSize), valImages.size());
            size_t currentBatchSize = batchEnd - i;
            
            // Prepare current validation batch
            std::vector<std::vector<float>> valBatchImages(currentBatchSize);
            std::vector<std::vector<float>> valBatchTargets(currentBatchSize);
            
            for (size_t j = 0; j < currentBatchSize; j++) {
                valBatchImages[j] = valImages[i + j];
                valBatchTargets[j] = valTargets[i + j];
            }
            
            // Forward pass on validation batch
            auto outputProbs = model.batchForward(valBatchImages);
            
            // Calculate metrics
            for (size_t j = 0; j < currentBatchSize; j++) {
                int predicted = std::max_element(outputProbs[j].begin(), outputProbs[j].end()) -
                               outputProbs[j].begin();
                int actual = valLabels[i + j];
                
                if (predicted == actual) {
                    valCorrect++;
                }
                
                // Calculate cross-entropy loss
                float sampleLoss = -std::log(std::max(outputProbs[j][actual], 1e-7f));
                valLoss += sampleLoss;
            }
        }
        
        float valAccuracy = static_cast<float>(valCorrect) / valImages.size();
        valLoss /= valImages.size();
        
        // Print epoch results
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << ", LR: " << learningRate
                  << ", Train Loss: " << trainLoss 
                  << ", Train Acc: " << trainAccuracy * 100 << "%" 
                  << ", Val Loss: " << valLoss
                  << ", Val Acc: " << valAccuracy * 100 << "%" << std::endl;
        
        // Early stopping check
        if (valAccuracy > bestValAccuracy) {
            bestValAccuracy = valAccuracy;
            patienceCounter = 0;
        } else {
            patienceCounter++;
            if (patienceCounter >= patience) {
                std::cout << "Early stopping at epoch " << epoch + 1 
                          << " (no improvement for " << patience << " epochs)" << std::endl;
                break;
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    std::cout << "Training completed in " << duration << " seconds" << std::endl;
    std::cout << "Best validation accuracy: " << bestValAccuracy * 100 << "%" << std::endl;
}

/**
 * Benchmark class for comparing original vs optimized implementations
 */
class PerformanceBenchmark {
private:
    struct BenchmarkResult {
        std::string name;
        double totalTimeSeconds;
        double samplesPerSecond;
        double speedupFactor;
        float finalAccuracy;
    };
    
    std::vector<BenchmarkResult> results;
    
public:
    // Add a benchmark result
    void addResult(const std::string& name, double totalTimeSeconds, 
                  size_t numSamples, float accuracy, double baselineTime = 0.0) {
        BenchmarkResult result;
        result.name = name;
        result.totalTimeSeconds = totalTimeSeconds;
        result.samplesPerSecond = numSamples / totalTimeSeconds;
        result.speedupFactor = (baselineTime > 0.0) ? baselineTime / totalTimeSeconds : 1.0;
        result.finalAccuracy = accuracy;
        
        results.push_back(result);
    }
    
    // Print results to console
    void printResults() const {
        std::cout << "\n============================================" << std::endl;
        std::cout << "           PERFORMANCE BENCHMARK            " << std::endl;
        std::cout << "============================================" << std::endl;
        
        std::cout << std::left << std::setw(25) << "Implementation" 
                  << std::setw(15) << "Time (sec)" 
                  << std::setw(15) << "Samples/sec" 
                  << std::setw(15) << "Speedup" 
                  << std::setw(15) << "Accuracy" << std::endl;
        
        std::cout << std::string(85, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(25) << result.name
                      << std::fixed << std::setprecision(2)
                      << std::setw(15) << result.totalTimeSeconds
                      << std::setw(15) << result.samplesPerSecond
                      << std::setw(15) << result.speedupFactor
                      << std::setw(15) << result.finalAccuracy * 100.0f << "%" << std::endl;
        }
        
        std::cout << "============================================" << std::endl;
    }
    
    // Save results to CSV file
    void saveResultsToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return;
        }
        
        file << "Implementation,Time (sec),Samples/sec,Speedup,Accuracy (%)\n";
        
        for (const auto& result : results) {
            file << result.name << ","
                 << std::fixed << std::setprecision(2) << result.totalTimeSeconds << ","
                 << result.samplesPerSecond << ","
                 << result.speedupFactor << ","
                 << result.finalAccuracy * 100.0f << "\n";
        }
        
        std::cout << "Benchmark results saved to " << filename << std::endl;
    }
};

/**
 * Original NeuralNetwork implementation (for comparison and compatibility)
 * This is kept for compatibility with existing code and benchmark comparison
 */
class NeuralNetwork {
private:
    // Network architecture
    int inputSize;
    int hiddenSize;
    int outputSize;
    
    // Network parameters
    std::vector<float> inputToHiddenWeights;
    std::vector<float> hiddenBiases;
    std::vector<float> hiddenToOutputWeights;
    std::vector<float> outputBiases;
    
    // Adam optimizer parameters
    std::vector<float> m_ih, v_ih; // Momentum and velocity for input-hidden weights
    std::vector<float> m_hb, v_hb; // Momentum and velocity for hidden biases
    std::vector<float> m_ho, v_ho; // Momentum and velocity for hidden-output weights
    std::vector<float> m_ob, v_ob; // Momentum and velocity for output biases
    float beta1 = 0.9;   // Exponential decay rate for first moment
    float beta2 = 0.999; // Exponential decay rate for second moment
    float epsilon = 1e-8; // Small constant for numerical stability
    int t = 0;          // Time step (updated during training)
    
    // ReLU activation function and its derivative
    static float relu(float x) {
        return std::max(0.0f, x);
    }
    
    static float reluDerivative(float x) {
        return x > 0 ? 1.0f : 0.0f;
    }
    
    // Softmax activation function (for multi-class output)
    std::vector<float> softmax(const std::vector<float>& x) const {
        std::vector<float> output(x.size());
        float maxVal = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < x.size(); i++) {
            output[i] = std::exp(x[i] - maxVal); // Subtract max for numerical stability
            sum += output[i];
        }
        
        for (size_t i = 0; i < x.size(); i++) {
            output[i] /= sum;
        }
        
        return output;
    }
    
public:
    NeuralNetwork(int inputSize, int hiddenSize = 128, int outputSize = 10) 
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize) {
        
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // He initialization for hidden layer
        float scale_ih = std::sqrt(2.0f / inputSize);
        std::normal_distribution<float> dist_ih(0, scale_ih);
        
        // Xavier initialization for output layer
        float scale_ho = std::sqrt(2.0f / (hiddenSize + outputSize));
        std::normal_distribution<float> dist_ho(0, scale_ho);
        
        // Initialize weights
        inputToHiddenWeights.resize(inputSize * hiddenSize);
        hiddenBiases.resize(hiddenSize);
        hiddenToOutputWeights.resize(hiddenSize * outputSize);
        outputBiases.resize(outputSize);
        
        // Initialize Adam optimizer parameters
        m_ih.resize(inputSize * hiddenSize, 0.0f);
        v_ih.resize(inputSize * hiddenSize, 0.0f);
        m_hb.resize(hiddenSize, 0.0f);
        v_hb.resize(hiddenSize, 0.0f);
        m_ho.resize(hiddenSize * outputSize, 0.0f);
        v_ho.resize(hiddenSize * outputSize, 0.0f);
        m_ob.resize(outputSize, 0.0f);
        v_ob.resize(outputSize, 0.0f);
        
        // Random initialization of weights
        for (int i = 0; i < inputSize * hiddenSize; i++) {
            inputToHiddenWeights[i] = dist_ih(gen);
        }
        
        for (int i = 0; i < hiddenSize; i++) {
            hiddenBiases[i] = 0.0f; // Initialize biases to zero
        }
        
        for (int i = 0; i < hiddenSize * outputSize; i++) {
            hiddenToOutputWeights[i] = dist_ho(gen);
        }
        
        for (int i = 0; i < outputSize; i++) {
            outputBiases[i] = 0.0f; // Initialize biases to zero
        }
    }
    
    // Forward pass (for a single digit)
    std::vector<float> forward(const std::vector<float>& input) const {
        // Hidden layer
        std::vector<float> hiddenActivations(hiddenSize);
        for (int j = 0; j < hiddenSize; j++) {
            float sum = hiddenBiases[j];
            for (int i = 0; i < inputSize; i++) {
                sum += input[i] * inputToHiddenWeights[i * hiddenSize + j];
            }
            hiddenActivations[j] = relu(sum);
        }
        
        // Output layer
        std::vector<float> outputActivations(outputSize);
        for (int k = 0; k < outputSize; k++) {
            float sum = outputBiases[k];
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenActivations[j] * hiddenToOutputWeights[j * outputSize + k];
            }
            outputActivations[k] = sum; // No activation before softmax
        }
        
        // Apply softmax to get probabilities
        return softmax(outputActivations);
    }
    
    // Returns the predicted digit (highest probability)
    int predict(const std::vector<float>& input) const {
        std::vector<float> probs = forward(input);
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }
    
    // Training step using Adam optimizer
    void train(const std::vector<float>& input, const std::vector<float>& target, float learningRate) {
        t++; // Increment time step
        
        // Forward pass with stored activations
        std::vector<float> hiddenLayerInput(hiddenSize, 0.0f);
        std::vector<float> hiddenLayerOutput(hiddenSize);
        std::vector<float> outputLayerInput(outputSize, 0.0f);
        std::vector<float> outputLayerOutput(outputSize);
        
        // Hidden layer forward
        for (int j = 0; j < hiddenSize; j++) {
            hiddenLayerInput[j] = hiddenBiases[j];
            for (int i = 0; i < inputSize; i++) {
                hiddenLayerInput[j] += input[i] * inputToHiddenWeights[i * hiddenSize + j];
            }
            hiddenLayerOutput[j] = relu(hiddenLayerInput[j]);
        }
        
        // Output layer forward
        for (int k = 0; k < outputSize; k++) {
            outputLayerInput[k] = outputBiases[k];
            for (int j = 0; j < hiddenSize; j++) {
                outputLayerInput[k] += hiddenLayerOutput[j] * hiddenToOutputWeights[j * outputSize + k];
            }
        }
        
        // Apply softmax
        outputLayerOutput = softmax(outputLayerInput);
        
        // Backpropagation - Calculate output layer gradients (cross-entropy with softmax)
        std::vector<float> outputGradients(outputSize);
        for (int k = 0; k < outputSize; k++) {
            outputGradients[k] = outputLayerOutput[k] - target[k];
        }
        
        // Calculate hidden layer gradients
        std::vector<float> hiddenGradients(hiddenSize, 0.0f);
        for (int j = 0; j < hiddenSize; j++) {
            float sum = 0.0f;
            for (int k = 0; k < outputSize; k++) {
                sum += outputGradients[k] * hiddenToOutputWeights[j * outputSize + k];
            }
            hiddenGradients[j] = sum * reluDerivative(hiddenLayerInput[j]);
        }
        
        // Calculate weight gradients for hidden-to-output weights
        std::vector<float> hoGradients(hiddenSize * outputSize);
        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                hoGradients[j * outputSize + k] = outputGradients[k] * hiddenLayerOutput[j];
            }
        }
        
        // Calculate weight gradients for input-to-hidden weights
        std::vector<float> ihGradients(inputSize * hiddenSize);
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                ihGradients[i * hiddenSize + j] = hiddenGradients[j] * input[i];
            }
        }
        
        // Apply Adam optimizer
        // Update input-hidden weights
        for (int i = 0; i < inputSize * hiddenSize; i++) {
            // Update biased first moment estimate
            m_ih[i] = beta1 * m_ih[i] + (1 - beta1) * ihGradients[i];
            // Update biased second moment estimate
            v_ih[i] = beta2 * v_ih[i] + (1 - beta2) * ihGradients[i] * ihGradients[i];
            
            // Compute bias-corrected first moment estimate
            float m_hat = m_ih[i] / (1 - std::pow(beta1, t));
            // Compute bias-corrected second moment estimate
            float v_hat = v_ih[i] / (1 - std::pow(beta2, t));
            
            // Update parameters
            inputToHiddenWeights[i] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
        
        // Update hidden biases
        for (int j = 0; j < hiddenSize; j++) {
            m_hb[j] = beta1 * m_hb[j] + (1 - beta1) * hiddenGradients[j];
            v_hb[j] = beta2 * v_hb[j] + (1 - beta2) * hiddenGradients[j] * hiddenGradients[j];
            
            float m_hat = m_hb[j] / (1 - std::pow(beta1, t));
            float v_hat = v_hb[j] / (1 - std::pow(beta2, t));
            
            hiddenBiases[j] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
        
        // Update hidden-output weights
        for (int i = 0; i < hiddenSize * outputSize; i++) {
            m_ho[i] = beta1 * m_ho[i] + (1 - beta1) * hoGradients[i];
            v_ho[i] = beta2 * v_ho[i] + (1 - beta2) * hoGradients[i] * hoGradients[i];
            
            float m_hat = m_ho[i] / (1 - std::pow(beta1, t));
            float v_hat = v_ho[i] / (1 - std::pow(beta2, t));
            
            hiddenToOutputWeights[i] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
        
        // Update output biases
        for (int k = 0; k < outputSize; k++) {
            m_ob[k] = beta1 * m_ob[k] + (1 - beta1) * outputGradients[k];
            v_ob[k] = beta2 * v_ob[k] + (1 - beta2) * outputGradients[k] * outputGradients[k];
            
            float m_hat = m_ob[k] / (1 - std::pow(beta1, t));
            float v_hat = v_ob[k] / (1 - std::pow(beta2, t));
            
            outputBiases[k] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
    
    // Save model to file
    bool saveModel(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return false;
        }
        
        // Write architecture
        file.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
        file.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(hiddenSize));
        file.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));
        
        // Write weights and biases
        file.write(reinterpret_cast<const char*>(inputToHiddenWeights.data()), inputToHiddenWeights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(hiddenBiases.data()), hiddenBiases.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(hiddenToOutputWeights.data()), hiddenToOutputWeights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(outputBiases.data()), outputBiases.size() * sizeof(float));
        
        return true;
    }
    
    // Load model from file
    bool loadModel(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error opening model file: " << filename << std::endl;
            return false;
        }
        
        // Read architecture
        int storedInputSize, storedHiddenSize, storedOutputSize;
        file.read(reinterpret_cast<char*>(&storedInputSize), sizeof(storedInputSize));
        file.read(reinterpret_cast<char*>(&storedHiddenSize), sizeof(storedHiddenSize));
        file.read(reinterpret_cast<char*>(&storedOutputSize), sizeof(storedOutputSize));
        
        // Check if architecture matches
        if (storedInputSize != inputSize || storedHiddenSize != hiddenSize || storedOutputSize != outputSize) {
            std::cerr << "Model architecture mismatch." << std::endl;
            return false;
        }
        
        // Read weights and biases
        file.read(reinterpret_cast<char*>(inputToHiddenWeights.data()), inputToHiddenWeights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(hiddenBiases.data()), hiddenBiases.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(hiddenToOutputWeights.data()), hiddenToOutputWeights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(outputBiases.data()), outputBiases.size() * sizeof(float));
        
        return true;
    }
};

/**
 * Data augmentation class to generate variations of digit images (original version)
 */
class DataAugmenter {
public:
    // Apply random transformations to an image
    static std::vector<float> augment(const std::vector<float>& image, int width = 28, int height = 28, float intensity = 0.5f) {
        std::vector<float> augmented = image;
        
        // Apply random transformations based on intensity
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        // Only apply augmentations with probability based on intensity
        if (dis(gen) < intensity) {
            // Apply elastic distortion
            applyElasticDistortion(augmented, width, height, 8.0f * intensity, 2.0f, gen);
        }
        
        if (dis(gen) < intensity) {
            // Apply small random rotation
            float angle = (dis(gen) * 2.0f - 1.0f) * 15.0f * intensity; // ±15 degrees max
            applyRotation(augmented, width, height, angle, gen);
        }
        
        if (dis(gen) < intensity) {
            // Apply small random shift
            int shiftX = static_cast<int>((dis(gen) * 2.0f - 1.0f) * 3.0f * intensity);
            int shiftY = static_cast<int>((dis(gen) * 2.0f - 1.0f) * 3.0f * intensity);
            applyShift(augmented, width, height, shiftX, shiftY);
        }
        
        if (dis(gen) < intensity) {
            // Apply small random scaling
            float scale = 1.0f + (dis(gen) * 2.0f - 1.0f) * 0.2f * intensity; // 0.8-1.2 scaling
            applyScaling(augmented, width, height, scale, gen);
        }
        
        // Apply random noise with a smaller probability
        if (dis(gen) < intensity * 0.5f) {
            applyNoise(augmented, 0.05f * intensity, gen);
        }
        
        return augmented;
    }
    
private:
    // Apply Gaussian noise to the image
    static void applyNoise(std::vector<float>& image, float stddev, std::mt19937& gen) {
        std::normal_distribution<float> noise(0.0f, stddev);
        for (auto& pixel : image) {
            pixel += noise(gen);
            pixel = std::max(0.0f, std::min(1.0f, pixel)); // Clamp to [0,1]
        }
    }
    
    // Apply random elastic distortion (simulates hand writing variations)
    static void applyElasticDistortion(std::vector<float>& image, int width, int height, 
                                      float alpha, float sigma, std::mt19937& gen) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        // Generate displacement fields
        std::vector<float> dx(width * height);
        std::vector<float> dy(width * height);
        
        // Generate random displacement fields
        for (int i = 0; i < width * height; i++) {
            dx[i] = dist(gen);
            dy[i] = dist(gen);
        }
        
        // Apply Gaussian filter to displacement fields
        std::vector<float> dxFiltered(width * height, 0.0f);
        std::vector<float> dyFiltered(width * height, 0.0f);
        
        // Simple Gaussian blur implementation
        int kernelSize = static_cast<int>(sigma * 3.0f) | 1; // Odd kernel size
        if (kernelSize < 3) kernelSize = 3;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sumX = 0.0f, sumY = 0.0f;
                float weightSum = 0.0f;
                
                for (int ky = -kernelSize/2; ky <= kernelSize/2; ky++) {
                    for (int kx = -kernelSize/2; kx <= kernelSize/2; kx++) {
                        int nx = x + kx;
                        int ny = y + ky;
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            float distance = std::sqrt(kx*kx + ky*ky);
                            float weight = std::exp(-(distance*distance) / (2.0f * sigma * sigma));
                            
                            sumX += dx[ny * width + nx] * weight;
                            sumY += dy[ny * width + nx] * weight;
                            weightSum += weight;
                        }
                    }
                }
                
                if (weightSum > 0) {
                    dxFiltered[y * width + x] = sumX / weightSum;
                    dyFiltered[y * width + x] = sumY / weightSum;
                }
            }
        }
        
        // Scale displacement fields
        for (int i = 0; i < width * height; i++) {
            dxFiltered[i] *= alpha;
            dyFiltered[i] *= alpha;
        }
        
        // Apply displacement fields
        std::vector<float> result(width * height, 0.0f);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float nx = x + dxFiltered[y * width + x];
                float ny = y + dyFiltered[y * width + x];
                
                // Bilinear interpolation
                if (nx >= 0 && nx < width - 1 && ny >= 0 && ny < height - 1) {
                    int x0 = static_cast<int>(std::floor(nx));
                    int y0 = static_cast<int>(std::floor(ny));
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    
                    float wx = nx - x0;
                    float wy = ny - y0;
                    
                    float val = (1-wx)*(1-wy) * image[y0 * width + x0] +
                               wx*(1-wy) * image[y0 * width + x1] +
                               (1-wx)*wy * image[y1 * width + x0] +
                               wx*wy * image[y1 * width + x1];
                    
                    result[y * width + x] = val;
                }
            }
        }
        
        image = result;
    }
    
    // Apply rotation to the image
    static void applyRotation(std::vector<float>& image, int width, int height, float angleDegrees, std::mt19937& gen) {
        float angleRad = angleDegrees * M_PI / 180.0f;
        float cosA = std::cos(angleRad);
        float sinA = std::sin(angleRad);
        
        std::vector<float> result(width * height, 0.0f);
        
        // Rotation around the center of the image
        float centerX = width / 2.0f;
        float centerY = height / 2.0f;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Translate coordinates to origin
                float xOrigin = x - centerX;
                float yOrigin = y - centerY;
                
                // Apply rotation
                float xRotated = xOrigin * cosA - yOrigin * sinA;
                float yRotated = xOrigin * sinA + yOrigin * cosA;
                
                // Translate back
                float xSource = xRotated + centerX;
                float ySource = yRotated + centerY;
                
                // Skip if outside the image
                if (xSource < 0 || xSource >= width - 1 || ySource < 0 || ySource >= height - 1) {
                    continue;
                }
                
                // Bilinear interpolation
                int x0 = static_cast<int>(std::floor(xSource));
                int y0 = static_cast<int>(std::floor(ySource));
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                float wx = xSource - x0;
                float wy = ySource - y0;
                
                // Calculate interpolated value
                float val = 0.0f;
                val += image[y0 * width + x0] * (1 - wx) * (1 - wy);
                val += image[y0 * width + x1] * wx * (1 - wy);
                val += image[y1 * width + x0] * (1 - wx) * wy;
                val += image[y1 * width + x1] * wx * wy;
                
                result[y * width + x] = val;
            }
        }
        
        image = result;
    }
    
    // Apply shift to the image
    static void applyShift(std::vector<float>& image, int width, int height, int shiftX, int shiftY) {
        std::vector<float> result(width * height, 0.0f);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Calculate source position
                int sourceX = x - shiftX;
                int sourceY = y - shiftY;
                
                // If within bounds, copy pixel
                if (sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height) {
                    result[y * width + x] = image[sourceY * width + sourceX];
                }
            }
        }
        
        image = result;
    }
    
    // Apply scaling to the image
    static void applyScaling(std::vector<float>& image, int width, int height, float scale, std::mt19937& gen) {
        std::vector<float> result(width * height, 0.0f);
        
        float centerX = width / 2.0f;
        float centerY = height / 2.0f;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Translate to origin, scale, and translate back
                float sourceX = centerX + (x - centerX) / scale;
                float sourceY = centerY + (y - centerY) / scale;
                
                // Skip if outside the image
                if (sourceX < 0 || sourceX >= width - 1 || sourceY < 0 || sourceY >= height - 1) {
                    continue;
                }
                
                // Bilinear interpolation
                int x0 = static_cast<int>(std::floor(sourceX));
                int y0 = static_cast<int>(std::floor(sourceY));
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                float wx = sourceX - x0;
                float wy = sourceY - y0;
                
                // Calculate interpolated value
                float val = 0.0f;
                val += image[y0 * width + x0] * (1 - wx) * (1 - wy);
                val += image[y0 * width + x1] * wx * (1 - wy);
                val += image[y1 * width + x0] * (1 - wx) * wy;
                val += image[y1 * width + x1] * wx * wy;
                
                result[y * width + x] = val;
            }
        }
        
        image = result;
    }
};

/**
 * Original training function (for comparison)
 */
void trainNetwork(NeuralNetwork& model, const MNISTDataset& trainData, const MNISTDataset& valData, 
                 int epochs, float initialLearningRate, int batchSize, float augmentationIntensity = 0.5f) {
    std::cout << "Training network with " << trainData.images.size() << " samples..." << std::endl;
    
    float learningRate = initialLearningRate;
    float bestValAccuracy = 0.0f;
    int patienceCounter = 0;
    const int patience = 5; // Early stopping patience
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // For confusion matrix during validation
    std::vector<std::vector<int>> confusionMatrix(10, std::vector<int>(10, 0));
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Apply learning rate decay
        if (epoch > 0 && epoch % 5 == 0) {
            learningRate *= 0.7f; // More aggressive decay
        }
        
        // Create mini-batches
        std::vector<std::pair<std::vector<std::vector<float>>, std::vector<int>>> miniBatches;
        
        // Create shuffled indices
        std::vector<size_t> indices(trainData.images.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Create batches
        for (size_t i = 0; i < trainData.images.size(); i += static_cast<size_t>(batchSize)) {
            std::vector<std::vector<float>> batchImages;
            std::vector<int> batchLabels;
            
            for (size_t j = i; j < i + static_cast<size_t>(batchSize) && j < trainData.images.size(); ++j) {
                batchImages.push_back(trainData.images[indices[j]]);
                batchLabels.push_back(trainData.labels[indices[j]]);
            }
            
            miniBatches.push_back({batchImages, batchLabels});
        }
        
        // Training metrics
        int trainCorrect = 0;
        float trainLoss = 0.0f;
        
        // Process each mini-batch
        for (size_t batchIdx = 0; batchIdx < miniBatches.size(); ++batchIdx) {
            const auto& batch = miniBatches[batchIdx];
            const auto& batchImages = batch.first;
            const auto& batchLabels = batch.second;
            
            for (size_t i = 0; i < batchImages.size(); ++i) {
                // Create one-hot encoded target
                std::vector<float> target(10, 0.0f);
                target[batchLabels[i]] = 1.0f;
                
                // Apply data augmentation with 50% probability
                std::vector<float> augmentedImage;
                if (augmentationIntensity > 0.0f && (static_cast<float>(rand()) / RAND_MAX) < 0.5f) {
                    augmentedImage = DataAugmenter::augment(batchImages[i], 28, 28, augmentationIntensity);
                } else {
                    augmentedImage = batchImages[i];
                }
                
                // Train on this example
                model.train(augmentedImage, target, learningRate);
                
                // Calculate loss and accuracy for this example
                std::vector<float> output = model.forward(augmentedImage);
                int predicted = std::max_element(output.begin(), output.end()) - output.begin();
                
                if (predicted == batchLabels[i]) {
                    trainCorrect++;
                }
                
                // Calculate cross-entropy loss
                float example_loss = -std::log(std::max(output[batchLabels[i]], 1e-7f));
                trainLoss += example_loss;
            }
            
            // Print progress for every 10% of batches
            if (batchIdx % std::max(1UL, miniBatches.size() / 10) == 0) {
                std::cout << "  Batch " << batchIdx << "/" << miniBatches.size() << " processed" << std::endl;
            }
        }
        
        // Calculate training metrics
        float trainAccuracy = static_cast<float>(trainCorrect) / trainData.images.size();
        trainLoss /= trainData.images.size();
        
        // Evaluate on validation set
        int valCorrect = 0;
        float valLoss = 0.0f;
        
        // Reset confusion matrix
        for (auto& row : confusionMatrix) {
            std::fill(row.begin(), row.end(), 0);
        }
        
        for (size_t i = 0; i < valData.images.size(); ++i) {
            std::vector<float> output = model.forward(valData.images[i]);
            int predicted = std::max_element(output.begin(), output.end()) - output.begin();
            
            // Update confusion matrix
            confusionMatrix[valData.labels[i]][predicted]++;
            
            if (predicted == valData.labels[i]) {
                valCorrect++;
            }
            
            // Calculate cross-entropy loss
            std::vector<float> target(10, 0.0f);
            target[valData.labels[i]] = 1.0f;
            float example_loss = -std::log(std::max(output[valData.labels[i]], 1e-7f));
            valLoss += example_loss;
        }
        
        float valAccuracy = static_cast<float>(valCorrect) / valData.images.size();
        valLoss /= valData.images.size();
        
        // Print epoch results
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << ", LR: " << learningRate
                  << ", Train Loss: " << trainLoss 
                  << ", Train Acc: " << trainAccuracy * 100 << "%" 
                  << ", Val Loss: " << valLoss
                  << ", Val Acc: " << valAccuracy * 100 << "%" << std::endl;
        
        // Every 5 epochs, print confusion matrix for certain digits
        if ((epoch + 1) % 5 == 0 || epoch == epochs - 1) {
            std::cout << "Confusion between digits 1, 5, and 8:" << std::endl;
            std::cout << "   |   1   |   5   |   8   |" << std::endl;
            std::cout << "---+-------+-------+-------+" << std::endl;
            std::cout << " 1 |  " << std::setw(4) << confusionMatrix[1][1] << " |  " 
                      << std::setw(4) << confusionMatrix[1][5] << " |  " 
                      << std::setw(4) << confusionMatrix[1][8] << " |" << std::endl;
            std::cout << " 5 |  " << std::setw(4) << confusionMatrix[5][1] << " |  " 
                      << std::setw(4) << confusionMatrix[5][5] << " |  " 
                      << std::setw(4) << confusionMatrix[5][8] << " |" << std::endl;
            std::cout << " 8 |  " << std::setw(4) << confusionMatrix[8][1] << " |  " 
                      << std::setw(4) << confusionMatrix[8][5] << " |  " 
                      << std::setw(4) << confusionMatrix[8][8] << " | "

                      << std::setw(4) << confusionMatrix[5][5] << " |  " 
                      << std::setw(4) << confusionMatrix[5][8] << " |" << std::endl;
            std::cout << " 8 |  " << std::setw(4) << confusionMatrix[8][1] << " |  " 
                      << std::setw(4) << confusionMatrix[8][5] << " |  " 
                      << std::setw(4) << confusionMatrix[8][8] << " |" << std::endl;
        }
        
        // Early stopping check
        if (valAccuracy > bestValAccuracy) {
            bestValAccuracy = valAccuracy;
            patienceCounter = 0;
        } else {
            patienceCounter++;
            if (patienceCounter >= patience) {
                std::cout << "Early stopping at epoch " << epoch + 1 
                          << " (no improvement for " << patience << " epochs)" << std::endl;
                break;
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    std::cout << "Training completed in " << duration << " seconds" << std::endl;
    std::cout << "Best validation accuracy: " << bestValAccuracy * 100 << "%" << std::endl;
}

/**
 * Function to convert from original MNISTDataset format to vector format used by optimized code
 */
void convertDatasetFormat(
    const MNISTDataset& dataset, 
    std::vector<std::vector<float>>& images,
    std::vector<int>& labels) {
    
    images = dataset.images;
    labels = dataset.labels;
}

/**
 * Function to run benchmarks comparing original and optimized implementations
 */
void runBenchmarks(
    const MNISTDataset& trainData,
    const MNISTDataset& valData,
    const MNISTDataset& testData,
    int inputSize,
    int hiddenSize,
    int outputSize,
    int epochs,
    float learningRate,
    int batchSize,
    float augmentationIntensity) {
    
    // Convert datasets to vector format for optimized version
    std::vector<std::vector<float>> trainImagesVec, valImagesVec, testImagesVec;
    std::vector<int> trainLabelsVec, valLabelsVec, testLabelsVec;
    
    convertDatasetFormat(trainData, trainImagesVec, trainLabelsVec);
    convertDatasetFormat(valData, valImagesVec, valLabelsVec);
    convertDatasetFormat(testData, testImagesVec, testLabelsVec);
    
    // Create benchmark instance
    PerformanceBenchmark benchmark;
    
    // Run original implementation benchmark
    {
        NeuralNetwork model(inputSize, hiddenSize, outputSize);
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Train using original implementation
        trainNetwork(model, trainData, valData, epochs, learningRate, batchSize, augmentationIntensity);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double>(endTime - startTime).count();
        
        // Calculate accuracy on test set
        int correct = 0;
        for (size_t i = 0; i < testData.images.size(); ++i) {
            int predicted = model.predict(testData.images[i]);
            if (predicted == testData.labels[i]) {
                correct++;
            }
        }
        
        float accuracy = static_cast<float>(correct) / testData.images.size();
        
        // Add result to benchmark
        benchmark.addResult("Original Implementation", totalTime, trainData.images.size() * epochs, accuracy);
        
        // Store baseline time for speedup calculation
        double baselineTime = totalTime;
        
        // Run optimized implementation with different configurations
        
        // 1. Basic optimization with vectorization
        {
            OptimizedNeuralNetwork model(inputSize, hiddenSize, outputSize);
            
            // Set single thread for this test
            omp_set_num_threads(1);
            
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // Train using optimized implementation with vectorization only
            trainOptimizedNetwork(
                model, trainImagesVec, trainLabelsVec, valImagesVec, valLabelsVec,
                epochs, learningRate, batchSize, augmentationIntensity
            );
            
            auto endTime = std::chrono::high_resolution_clock::now();
            double totalTime = std::chrono::duration<double>(endTime - startTime).count();
            
            // Calculate accuracy on test set
            int correct = 0;
            for (size_t i = 0; i < testImagesVec.size(); ++i) {
                int predicted = model.predict(testImagesVec[i]);
                if (predicted == testLabelsVec[i]) {
                    correct++;
                }
            }
            
            float accuracy = static_cast<float>(correct) / testImagesVec.size();
            
            // Add result to benchmark
            benchmark.addResult("Vectorized (Single Thread)", totalTime, 
                              trainImagesVec.size() * epochs, accuracy, baselineTime);
        }
        
        // 2. Full optimization with multi-threading
        {
            OptimizedNeuralNetwork model(inputSize, hiddenSize, outputSize);
            
            // Set max threads
            int numThreads = std::thread::hardware_concurrency();
            omp_set_num_threads(numThreads);
            
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // Train using fully optimized implementation
            trainOptimizedNetwork(
                model, trainImagesVec, trainLabelsVec, valImagesVec, valLabelsVec,
                epochs, learningRate, batchSize, augmentationIntensity
            );
            
            auto endTime = std::chrono::high_resolution_clock::now();
            double totalTime = std::chrono::duration<double>(endTime - startTime).count();
            
            // Calculate accuracy on test set
            int correct = 0;
            for (size_t i = 0; i < testImagesVec.size(); ++i) {
                int predicted = model.predict(testImagesVec[i]);
                if (predicted == testLabelsVec[i]) {
                    correct++;
                }
            }
            
            float accuracy = static_cast<float>(correct) / testImagesVec.size();
            
            // Add result to benchmark
            benchmark.addResult("Fully Optimized (Multi-Thread)", totalTime, 
                              trainImagesVec.size() * epochs, accuracy, baselineTime);
        }
        
        // 3. Test different batch sizes
        if (batchSize < 256) {  // Only run if original batch size is reasonably small
            OptimizedNeuralNetwork model(inputSize, hiddenSize, outputSize);
            
            int largerBatchSize = std::min(256, batchSize * 4);  // Try a larger batch size
            
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // Train using larger batch size
            trainOptimizedNetwork(
                model, trainImagesVec, trainLabelsVec, valImagesVec, valLabelsVec,
                epochs, learningRate, largerBatchSize, augmentationIntensity
            );
            
            auto endTime = std::chrono::high_resolution_clock::now();
            double totalTime = std::chrono::duration<double>(endTime - startTime).count();
            
            // Calculate accuracy on test set
            int correct = 0;
            for (size_t i = 0; i < testImagesVec.size(); ++i) {
                int predicted = model.predict(testImagesVec[i]);
                if (predicted == testLabelsVec[i]) {
                    correct++;
                }
            }
            
            float accuracy = static_cast<float>(correct) / testImagesVec.size();
            
            // Add result to benchmark
            benchmark.addResult(
                "Optimized (Batch Size " + std::to_string(largerBatchSize) + ")",
                totalTime, trainImagesVec.size() * epochs, accuracy, baselineTime
            );
        }
        
        // Print and save benchmark results
        benchmark.printResults();
        benchmark.saveResultsToCSV("mnist_optimization_benchmark.csv");
    }
}

/**
 * Enhanced preprocessing for custom handwritten digits
 */
std::vector<float> preprocessImage(const std::string& imagePath, float mean, float stdDev) {
    int width, height, channels;
    unsigned char* imageData = stbi_load(imagePath.c_str(), &width, &height, &channels, 0);
    
    if (!imageData) {
        std::cerr << "Error loading image: " << imagePath << std::endl;
        return {};
    }
    
    // Create output vector
    std::vector<float> grayscaleImage(width * height, 0.0f);
    std::vector<float> processedImage(28 * 28, 0.0f);
    
    // Convert to grayscale and calculate center of mass
    float totalMass = 0.0f;
    float centerX = 0.0f;
    float centerY = 0.0f;
    float maxPixelValue = 0.0f;
    
    // Step 1: Convert to grayscale
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            
            // Convert to grayscale
            float gray;
            if (channels >= 3) {
                gray = 0.299f * imageData[idx] + 0.587f * imageData[idx + 1] + 0.114f * imageData[idx + 2];
            } else {
                gray = imageData[idx];
            }
            
            // Invert if needed (assuming black digit on white background)
            gray = 255.0f - gray;
            
            // Store and track max value
            grayscaleImage[y * width + x] = gray;
            maxPixelValue = std::max(maxPixelValue, gray);
        }
    }
    
    // Normalize to ensure proper thresholding
    if (maxPixelValue > 0) {
        for (auto& pixel : grayscaleImage) {
            pixel /= maxPixelValue;
            pixel = pixel > 0.2f ? pixel : 0.0f; // Simple thresholding to remove noise
        }
    }
    
    // Recalculate center of mass with thresholded image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float gray = grayscaleImage[y * width + x];
            totalMass += gray;
            centerX += x * gray;
            centerY += y * gray;
        }
    }
    
    // Calculate center of mass
    if (totalMass > 0) {
        centerX /= totalMass;
        centerY /= totalMass;
    } else {
        centerX = width / 2.0f;
        centerY = height / 2.0f;
    }
    
    // Calculate the bounding box of the digit
    int minX = width, minY = height, maxX = 0, maxY = 0;
    float threshold = 0.1f; // Threshold for considering a pixel part of the digit
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (grayscaleImage[y * width + x] > threshold) {
                minX = std::min(minX, x);
                minY = std::min(minY, y);
                maxX = std::max(maxX, x);
                maxY = std::max(maxY, y);
            }
        }
    }
    
    // Calculate scale factor to fit the digit in a slightly smaller area (to leave margin)
    float digitWidth = maxX - minX + 1;
    float digitHeight = maxY - minY + 1;
    float targetSize = 20.0f; // Target size with margins
    float scaleX = targetSize / digitWidth;
    float scaleY = targetSize / digitHeight;
    float scale = std::min(scaleX, scaleY);
    
    // Ensure we don't upscale too much for very small inputs
    scale = std::min(scale, 5.0f);
    
    // Resize and center the digit based on center of mass and bounding box
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            // Map to source coordinates with centering adjustment
            // Center in the target image and apply scaling
            float srcX = (x - 14) / scale + centerX;
            float srcY = (y - 14) / scale + centerY;
            
            // Bilinear interpolation
            int x0 = static_cast<int>(std::floor(srcX));
            int y0 = static_cast<int>(std::floor(srcY));
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float wx = srcX - x0;
            float wy = srcY - y0;
            
            float val = 0.0f;
            float weightSum = 0.0f;
            
            // Accumulate values from the 4 surrounding pixels
            for (int yy = y0; yy <= y1; yy++) {
                for (int xx = x0; xx <= x1; xx++) {
                    if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                        float weight = (1 - std::abs(xx - srcX)) * (1 - std::abs(yy - srcY));
                        val += grayscaleImage[yy * width + xx] * weight;
                        weightSum += weight;
                    }
                }
            }
            
            // Normalize by weight sum
            if (weightSum > 0) {
                val /= weightSum;
            }
            
            // Store normalized value
            processedImage[y * 28 + x] = val;
        }
    }
    
    // Free the loaded image
    stbi_image_free(imageData);
    
    // Apply the same normalization as used in training
    normalizeImage(processedImage, mean, stdDev);
    
    return processedImage;
}

/**
 * Recognize a single digit from an image file
 */
void recognizeDigit(OptimizedNeuralNetwork& model, const std::string& imagePath, float mean, float stdDev) {
    // Preprocess the image
    std::vector<float> image = preprocessImage(imagePath, mean, stdDev);
    if (image.empty()) {
        std::cout << "Failed to process image" << std::endl;
        return;
    }
    
    // Run inference
    std::vector<float> probabilities = model.forward(image);
    
    // Print probabilities for all digits
    std::cout << "Confidence scores:" << std::endl;
    for (int digit = 0; digit < 10; digit++) {
        std::cout << "  Digit " << digit << ": " << std::fixed << std::setprecision(4) 
                  << probabilities[digit] << std::endl;
    }
    
    // Find the predicted digit
    int predictedDigit = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
    std::cout << "RECOGNIZED DIGIT: " << predictedDigit << std::endl;
    
    // Print top-3 digits for better analysis
    std::vector<std::pair<float, int>> scores;
    for (int i = 0; i < 10; i++) {
        scores.push_back({probabilities[i], i});
    }
    
    std::sort(scores.begin(), scores.end(), std::greater<>());
    
    std::cout << "Top 3 predictions:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  " << i+1 << ". Digit " << scores[i].second 
                  << " with confidence " << scores[i].first * 100 << "%" << std::endl;
    }
}

// Function to convert big-endian to little-endian
uint32_t swapEndian(uint32_t value) {
    return ((value & 0xff) << 24) | 
           ((value & 0xff00) << 8) |
           ((value & 0xff0000) >> 8) | 
           ((value & 0xff000000) >> 24);
}

// Normalize dataset
std::pair<float, float> normalizeDataset(std::vector<std::vector<float>>& images) {
    // Calculate mean pixel value
    float sum = 0.0f;
    size_t totalPixels = 0;
    
    for (const auto& image : images) {
        sum += std::accumulate(image.begin(), image.end(), 0.0f);
        totalPixels += image.size();
    }
    
    float mean = sum / totalPixels;
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (const auto& image : images) {
        for (float pixel : image) {
            float diff = pixel - mean;
            variance += diff * diff;
        }
    }
    
    float stdDev = std::sqrt(variance / totalPixels);
    
    // Avoid division by zero
    if (stdDev < 1e-5f) stdDev = 1.0f;
    
    // Normalize each image: (x - mean) / stdDev
    for (auto& image : images) {
        for (auto& pixel : image) {
            pixel = (pixel - mean) / stdDev;
        }
    }
    
    std::cout << "Dataset normalized: mean=" << mean << ", stdDev=" << stdDev << std::endl;
    return {mean, stdDev};
}

// Apply normalization to a single image
void normalizeImage(std::vector<float>& image, float mean, float stdDev) {
    for (auto& pixel : image) {
        pixel = (pixel - mean) / stdDev;
    }
}

// Read MNIST dataset
MNISTDataset readMNISTDataset(const std::string& imagesFile, const std::string& labelsFile) {
    MNISTDataset dataset;

    // Open files
    std::ifstream imagesStream(imagesFile, std::ios::binary);
    if (!imagesStream) {
        std::cerr << "Error opening images file: " << imagesFile << std::endl;
        return dataset;
    }

    std::ifstream labelsStream(labelsFile, std::ios::binary);
    if (!labelsStream) {
        std::cerr << "Error opening labels file: " << labelsFile << std::endl;
        return dataset;
    }

    // Read headers
    uint32_t magicNumber, numImages, numRows, numCols;
    imagesStream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    imagesStream.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    imagesStream.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    imagesStream.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
    
    magicNumber = swapEndian(magicNumber);
    numImages = swapEndian(numImages);
    numRows = swapEndian(numRows);
    numCols = swapEndian(numCols);

    uint32_t labelsMagicNumber, numLabels;
    labelsStream.read(reinterpret_cast<char*>(&labelsMagicNumber), sizeof(labelsMagicNumber));
    labelsStream.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    
    labelsMagicNumber = swapEndian(labelsMagicNumber);
    numLabels = swapEndian(numLabels);

    std::cout << "Loading MNIST dataset with " << numImages << " images (" 
              << numRows << "x" << numCols << " pixels)" << std::endl;

    // Verify headers
    if (numImages != numLabels) {
        std::cerr << "Error: Number of images and labels don't match!" << std::endl;
        return dataset;
    }
    if (magicNumber != 2051 || labelsMagicNumber != 2049) {
        std::cerr << "Error: Invalid magic numbers" << std::endl;
        return dataset;
    }

    // Allocate memory
    const int imageSize = numRows * numCols;
    dataset.images.resize(numImages, std::vector<float>(imageSize));
    dataset.labels.resize(numImages);

    // Read data
    for (uint32_t i = 0; i < numImages; ++i) {
        // Read image
        for (int j = 0; j < imageSize; ++j) {
            unsigned char pixel;
            imagesStream.read(reinterpret_cast<char*>(&pixel), 1);
            dataset.images[i][j] = static_cast<float>(pixel) / 255.0f;
        }

        // Read label
        unsigned char label;
        labelsStream.read(reinterpret_cast<char*>(&label), 1);
        dataset.labels[i] = static_cast<int>(label);
    }

    std::cout << "Successfully loaded " << numImages << " images and labels." << std::endl;
    return dataset;
}

// Create train/validation split
std::pair<MNISTDataset, MNISTDataset> createTrainValSplit(const MNISTDataset& trainData, float valRatio) {
    MNISTDataset trainSet, valSet;
    
    // Create shuffled indices
    std::vector<size_t> indices(trainData.images.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Calculate split sizes
    size_t valSize = static_cast<size_t>(trainData.images.size() * valRatio);
    size_t trainSize = trainData.images.size() - valSize;
    
    // Resize datasets
    trainSet.images.resize(trainSize);
    trainSet.labels.resize(trainSize);
    valSet.images.resize(valSize);
    valSet.labels.resize(valSize);
    
    // Fill train set
    for (size_t i = 0; i < trainSize; ++i) {
        trainSet.images[i] = trainData.images[indices[i]];
        trainSet.labels[i] = trainData.labels[indices[i]];
    }
    
    // Fill validation set
    for (size_t i = 0; i < valSize; ++i) {
        valSet.images[i] = trainData.images[indices[trainSize + i]];
        valSet.labels[i] = trainData.labels[indices[trainSize + i]];
    }
    
    std::cout << "Split dataset into " << trainSize << " training samples and " 
              << valSize << " validation samples" << std::endl;
              
    return {trainSet, valSet};
}

/**
 * Main function with optimized MNIST neural network implementation
 */
int main(int argc, char* argv[]) {
    // Default file paths for MNIST dataset
    std::string trainImagesFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-images.idx3-ubyte";
    std::string trainLabelsFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-labels.idx1-ubyte";
    std::string testImagesFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-images.idx3-ubyte";
    std::string testLabelsFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-labels.idx1-ubyte";
    
    // Parse command line options
    bool benchmarkMode = false;
    bool useOptimizedTraining = true;
    bool trainMode = false;
       // Training parameters
    int epochs = 30;               // More epochs for better convergence
    float learningRate = 0.001f;   // Lower learning rate for Adam optimizer
    int batchSize = 256;            // Mini-batch size increased from 64
    float augmentationIntensity = 0.5f; // Intensity of data augmentation reduced from 0.5
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--benchmark") {
            benchmarkMode = true;
        } else if (arg == "--legacy") {
            useOptimizedTraining = false;
        }
        else if (arg == "--train")
        {
            trainMode=true; 
        }
        else if (arg == "--epoch")
        {
            epochs = std::stoi(argv[i+1]);
        }
        else if (arg == "--augmentationIntensity")
        {
            augmentationIntensity = std::stof(argv[i+1]);
        }
    }
    
    // Set number of OpenMP threads for parallel processing
    int numThreads = std::thread::hardware_concurrency();
    omp_set_num_threads(numThreads);
    std::cout << "Using " << numThreads << " OpenMP threads for parallel processing" << std::endl;
    
    // Read datasets
    std::cout << "Reading MNIST training dataset..." << std::endl;
    MNISTDataset trainDataFull = readMNISTDataset(trainImagesFile, trainLabelsFile);
    if (trainDataFull.images.empty()) {
        std::cerr << "Failed to load training data." << std::endl;
        return 1;
    }

    std::cout << "Reading MNIST test dataset..." << std::endl;
    MNISTDataset testData = readMNISTDataset(testImagesFile, testLabelsFile);
    if (testData.images.empty()) {
        std::cerr << "Failed to load test data." << std::endl;
        return 1;
    }
    
    // Create train/validation split
    auto [trainData, valData] = createTrainValSplit(trainDataFull);
    
    // Normalize datasets
    std::cout << "Normalizing datasets..." << std::endl;
    auto [mean, stdDev] = normalizeDataset(trainData.images);
    normalizeDataset(valData.images);
    normalizeDataset(testData.images);
    
    // Create output directory for model files
    std::filesystem::create_directory("models");

    // Initialize neural network
    const int inputSize = 28 * 28;  // MNIST image size
    const int hiddenSize = 128;     // Size of hidden layer
    const int outputSize = 10;      // 10 digits
    
 
    
    // Model file path
    std::string modelPath = "models/mnist_mlp.model";
    
    // Run benchmark mode if requested
    if (benchmarkMode) {
        std::cout << "\n=== Running Performance Benchmarks ===\n" << std::endl;
        runBenchmarks(
            trainData, valData, testData,
            inputSize, hiddenSize, outputSize,
            5, // Use fewer epochs for benchmarking
            learningRate, batchSize, augmentationIntensity
        );
        return 0;
    }
    
    // Convert datasets to vector format for optimized version
    std::vector<std::vector<float>> trainImagesVec, valImagesVec, testImagesVec;
    std::vector<int> trainLabelsVec, valLabelsVec, testLabelsVec;
    
    convertDatasetFormat(trainData, trainImagesVec, trainLabelsVec);
    convertDatasetFormat(valData, valImagesVec, valLabelsVec);
    convertDatasetFormat(testData, testImagesVec, testLabelsVec);
    
    if (trainMode) {
        if (useOptimizedTraining) {
            // Use optimized implementation
            std::cout << "\n=== Training Neural Network with Optimized Algorithm ===\n" << std::endl;
            
            OptimizedNeuralNetwork model(inputSize, hiddenSize, outputSize);
            
            // Train with optimized batch processing
            trainOptimizedNetwork(
                model, trainImagesVec, trainLabelsVec, valImagesVec, valLabelsVec, 
                epochs, learningRate, batchSize, augmentationIntensity
            );
            
            // Evaluate on test set
            std::cout << "\n=== Evaluating on Test Set ===\n" << std::endl;
            
            int correct = 0;
            std::vector<std::vector<int>> confusionMatrix(10, std::vector<int>(10, 0));
            
            // Process test set in batches for better performance
            const int testBatchSize = 100;
            for (size_t i = 0; i < testImagesVec.size(); i += static_cast<size_t>(testBatchSize)) {
                size_t batchEnd = std::min(i + static_cast<size_t>(testBatchSize), testImagesVec.size());
                
                for (size_t j = i; j < batchEnd; j++) {
                    int predicted = model.predict(testImagesVec[j]);
                    int actual = testLabelsVec[j];
                    
                    confusionMatrix[actual][predicted]++;
                    
                    if (predicted == actual) {
                        correct++;
                    }
                }
            }
            
            float accuracy = static_cast<float>(correct) / testImagesVec.size();
            std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
            
            // Print full confusion matrix
            std::cout << "\nConfusion Matrix:" << std::endl;
            std::cout << "    ";
            for (int i = 0; i < 10; ++i) {
                std::cout << std::setw(5) << i;
            }
            std::cout << " <- Predicted" << std::endl;
            
            for (int i = 0; i < 10; ++i) {
                std::cout << std::setw(3) << i << " |";
                for (int j = 0; j < 10; ++j) {
                    std::cout << std::setw(5) << confusionMatrix[i][j];
                }
                std::cout << std::endl;
            }
            std::cout << "^ Actual" << std::endl;
            
            // Save model - implement your own saving logic
            if (model.saveModel(modelPath)) {
                std::cout << "Saved model to " << modelPath << std::endl;
            } else {
                std::cerr << "Failed to save model" << std::endl;
            }
            
            // Save normalization parameters
            std::ofstream normFile("models/normalization.txt");
            if (normFile) {
                normFile << mean << " " << stdDev;
                std::cout << "\nSaved normalization parameters for inference" << std::endl;
            }
        } 
        else {
            // Use original implementation
            std::cout << "\n=== Training Neural Network with Original Algorithm ===\n" << std::endl;
            
            NeuralNetwork model(inputSize, hiddenSize, outputSize);
            
            // Train with original implementation
            trainNetwork(model, trainData, valData, epochs, learningRate, batchSize, augmentationIntensity);
            
            // Evaluate on test set
            std::cout << "\n=== Evaluating on Test Set ===\n" << std::endl;
            
            int correct = 0;
            std::vector<std::vector<int>> confusionMatrix(10, std::vector<int>(10, 0));
            
            for (size_t i = 0; i < testData.images.size(); ++i) {
                int predicted = model.predict(testData.images[i]);
                int actual = testData.labels[i];
                
                confusionMatrix[actual][predicted]++;
                
                if (predicted == actual) {
                    correct++;
                }
            }
            
            float accuracy = static_cast<float>(correct) / testData.images.size();
            std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
            
            // Save model
            if (model.saveModel(modelPath)) {
                std::cout << "Saved model to " << modelPath << std::endl;
            } else {
                std::cerr << "Failed to save model" << std::endl;
            }
            
            // Save normalization parameters
            std::ofstream normFile("models/normalization.txt");
            if (normFile) {
                normFile << mean << " " << stdDev;
                std::cout << "\nSaved normalization parameters for inference" << std::endl;
            }
        }
    } 
    else {
        // Inference mode - load pre-trained model
        std::cout << "\n=== Loading Pre-trained Model ===\n" << std::endl;
        
        // Load normalization parameters
        float loadedMean = 0.0f, loadedStdDev = 1.0f;
        std::ifstream normFile("models/normalization.txt");
        if (normFile) {
            normFile >> loadedMean >> loadedStdDev;
            std::cout << "Loaded normalization parameters: mean=" << loadedMean << ", stdDev=" << loadedStdDev << std::endl;
        } else {
            std::cout << "No normalization file found, using defaults" << std::endl;
        }
        
        // Choose which model type to load based on command line arg
        if (useOptimizedTraining) {
            OptimizedNeuralNetwork model(inputSize, hiddenSize, outputSize);
            
            // Load model
            if (!model.loadModel(modelPath)) {
                std::cerr << "Failed to load model. Please run in training mode first." << std::endl;
                return 1;
            }
            std::cout << "Successfully loaded model from " << modelPath << std::endl;
            
            // Process custom images if provided
            if (argc >= 2) {
                std::cout << "\n=== Recognizing Custom Handwritten Digits ===\n" << std::endl;
                
                for (int i = 1; i < argc; i++) {
                    std::string arg = argv[i];
                    
                    // Skip command line options
                    if (arg.substr(0, 2) == "--") {
                        continue;
                    }
                    
                    // Process the image
                    std::string imagePath = arg;
                    std::cout << "\nProcessing image: " << imagePath << std::endl;
                    recognizeDigit(model, imagePath, loadedMean, loadedStdDev);
                }
            }
        } else {
            NeuralNetwork model(inputSize, hiddenSize, outputSize);
            
            // Load model
            if (!model.loadModel(modelPath)) {
                std::cerr << "Failed to load model. Please run in training mode first." << std::endl;
                return 1;
            }
            std::cout << "Successfully loaded model from " << modelPath << std::endl;
            
            // Original recognition code would go here
        }
    }
    
    std::cout << "\nTo recognize your own handwritten digits:" << std::endl;
    std::cout << "1. Run in training mode first" << std::endl;
    std::cout << "2. Create images of handwritten digits (black on white background)" << std::endl;
    std::cout << "3. Run: ./main your_image1.png your_image2.jpg ..." << std::endl;
    std::cout << "4. Use --legacy flag to use original implementation" << std::endl;
    std::cout << "5. Use --benchmark to compare performance" << std::endl;
    std::cout << "6. Use --train to compare performance" << std::endl;
    return 0;
}