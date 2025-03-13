/**
 * @file activation_functions.cu
 * @brief Implementation of various neural network activation functions using CUDA
 * 
 * This file implements and benchmarks the following activation functions:
 * - ReLU (Rectified Linear Unit)
 * - LeakyReLU
 * - ELU (Exponential Linear Unit)
 * - GELU (Gaussian Error Linear Unit)
 * - Softmax
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <float.h>
 #include <cuda_runtime.h>
 #include <chrono>
 #include <iostream>
 #include <vector>
 #include <algorithm>
 #include <stdexcept>
 #include <random>
 
 /**
  * @brief Error checking macro for CUDA API calls
  * This macro wraps CUDA API calls and checks for errors, printing a detailed error message and exiting if an error occurs
  */
 #define CUDA_CHECK(call) do { \
     cudaError_t error = call; \
     if (error != cudaSuccess) { \
         fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                 cudaGetErrorString(error)); \
         exit(EXIT_FAILURE); \
     } \
 } while(0)
 
 // ==============================
 // CUDA Kernel Implementations
 // ==============================
 
 /**
  * @brief ReLU activation function kernel
  * f(x) = max(0, x)
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param size Number of elements in the arrays
  * 
  * Time Complexity: O(n) where n is the number of elements
  * Space Complexity: O(1) per thread
  */
 __global__ void reluKernel(const float* input, float* output, const int size) {
     // Calculate global thread ID
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Check if thread is within bounds
     if (idx < size) {
         // Apply ReLU activation
         output[idx] = fmaxf(0.0f, input[idx]);
     }
 }
 
 /**
  * @brief LeakyReLU activation function kernel
  * f(x) = x if x > 0, else f(x) = alpha * x
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param size Number of elements in the arrays
  * @param alpha Slope for negative values (typically small, e.g., 0.01)
  * 
  * Time Complexity: O(n) where n is the number of elements
  * Space Complexity: O(1) per thread
  */
 __global__ void leakyReluKernel(const float* input, float* output, const int size, const float alpha) {
     // Calculate global thread ID
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Check if thread is within bounds
     if (idx < size) {
         // Apply Leaky ReLU activation
         output[idx] = input[idx] > 0.0f ? input[idx] : alpha * input[idx];
     }
 }
 
 /**
  * @brief ELU (Exponential Linear Unit) activation function kernel
  * f(x) = x if x > 0, else f(x) = alpha * (exp(x) - 1)
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param size Number of elements in the arrays
  * @param alpha Scale for the exponential part (typically 1.0)
  * 
  * Time Complexity: O(n) where n is the number of elements
  * Space Complexity: O(1) per thread
  */
 __global__ void eluKernel(const float* input, float* output, const int size, const float alpha) {
     // Calculate global thread ID
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Check if thread is within bounds
     if (idx < size) {
         // Apply ELU activation
         output[idx] = input[idx] > 0.0f ? input[idx] : alpha * (expf(input[idx]) - 1.0f);
     }
 }
 
 /**
  * @brief GELU (Gaussian Error Linear Unit) activation function kernel
  * f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param size Number of elements in the arrays
  * 
  * Time Complexity: O(n) where n is the number of elements
  * Space Complexity: O(1) per thread
  */
 __global__ void geluKernel(const float* input, float* output, const int size) {
     // Calculate global thread ID
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Check if thread is within bounds
     if (idx < size) {
         // Constants for GELU approximation
         const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
         const float coeff = 0.044715f;
         
         // Apply GELU activation using approximation
         float x = input[idx];
         float x3 = x * x * x;
         output[idx] = 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coeff * x3)));
     }
 }
 
 /**
  * @brief Softmax activation function kernel (for a batch of vectors)
  * f(x_i) = exp(x_i) / sum(exp(x_j)) for all j in the same vector
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param batch_size Number of vectors (batches)
  * @param dim Dimension of each vector
  * 
  * Time Complexity: O(batch_size * dim)
  * Space Complexity: O(1) per thread plus O(batch_size * blockDim.x) for shared memory
  * 
  * Note: This implementation uses shared memory for reduction operations to improve efficiency
  */
 __global__ void softmaxKernel(const float* input, float* output, const int batch_size, const int dim) {
     // Define shared memory for reduction operations
     // We need shared memory for finding the maximum value and computing the sum of exponents
     extern __shared__ float shared_mem[];
     
     // One thread block processes one batch
     int batch_idx = blockIdx.x;
     
     // Bounds check for batch index
     if (batch_idx >= batch_size) return;
     
     // Pointer to the current batch
     const float* batch_input = input + batch_idx * dim;
     float* batch_output = output + batch_idx * dim;
     
     // Step 1: Find maximum value for numerical stability (to prevent overflow in exp)
     float max_val = -FLT_MAX;
     for (int i = threadIdx.x; i < dim; i += blockDim.x) {
         max_val = fmaxf(max_val, batch_input[i]);
     }
     
     // Each thread stores its local maximum in shared memory
     shared_mem[threadIdx.x] = max_val;
     __syncthreads();  // Synchronize all threads in the block
     
     // Parallel reduction to find global maximum across all threads
     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
         if (threadIdx.x < stride) {
             shared_mem[threadIdx.x] = fmaxf(shared_mem[threadIdx.x], shared_mem[threadIdx.x + stride]);
         }
         __syncthreads();  // Synchronize all threads in the block
     }
     
     // All threads now have access to the maximum value
     max_val = shared_mem[0];
     
     // Step 2: Compute sum of exp(x - max_val) for normalization
     float sum = 0.0f;
     for (int i = threadIdx.x; i < dim; i += blockDim.x) {
         sum += expf(batch_input[i] - max_val);
     }
     
     // Store local sum in shared memory
     shared_mem[threadIdx.x] = sum;
     __syncthreads();  // Synchronize all threads in the block
     
     // Parallel reduction to compute global sum
     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
         if (threadIdx.x < stride) {
             shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
         }
         __syncthreads();  // Synchronize all threads in the block
     }
     
     // All threads now have access to the sum
     sum = shared_mem[0];
     
     // Step 3: Compute softmax values
     for (int i = threadIdx.x; i < dim; i += blockDim.x) {
         batch_output[i] = expf(batch_input[i] - max_val) / sum;
     }
 }
 
 // ==============================
 // Host Wrapper Functions
 // ==============================
 
 /**
  * @brief Host wrapper for ReLU activation function
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param size Number of elements in the arrays
  */
 void relu(const float* input, float* output, const int size) {
     // Calculate grid dimensions
     int blockSize = 256;  // Threads per block
     int numBlocks = (size + blockSize - 1) / blockSize;  // Ceiling division
     
     // Launch kernel
     reluKernel<<<numBlocks, blockSize>>>(input, output, size);
     
     // Check for kernel launch errors
     CUDA_CHECK(cudaPeekAtLastError());
     CUDA_CHECK(cudaDeviceSynchronize());
 }
 
 /**
  * @brief Host wrapper for Leaky ReLU activation function
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param size Number of elements in the arrays
  * @param alpha Slope for negative values (default: 0.01)
  */
 void leakyRelu(const float* input, float* output, const int size, const float alpha = 0.01f) {
     // Calculate grid dimensions
     int blockSize = 256;
     int numBlocks = (size + blockSize - 1) / blockSize;
     
     // Launch kernel
     leakyReluKernel<<<numBlocks, blockSize>>>(input, output, size, alpha);
     
     // Check for kernel launch errors
     CUDA_CHECK(cudaPeekAtLastError());
     CUDA_CHECK(cudaDeviceSynchronize());
 }
 
 /**
  * @brief Host wrapper for ELU activation function
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param size Number of elements in the arrays
  * @param alpha Scale for the exponential part (default: 1.0)
  */
 void elu(const float* input, float* output, const int size, const float alpha = 1.0f) {
     // Calculate grid dimensions
     int blockSize = 256;
     int numBlocks = (size + blockSize - 1) / blockSize;
     
     // Launch kernel
     eluKernel<<<numBlocks, blockSize>>>(input, output, size, alpha);
     
     // Check for kernel launch errors
     CUDA_CHECK(cudaPeekAtLastError());
     CUDA_CHECK(cudaDeviceSynchronize());
 }
 
 /**
  * @brief Host wrapper for GELU activation function
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param size Number of elements in the arrays
  */
 void gelu(const float* input, float* output, const int size) {
     // Calculate grid dimensions
     int blockSize = 256;
     int numBlocks = (size + blockSize - 1) / blockSize;
     
     // Launch kernel
     geluKernel<<<numBlocks, blockSize>>>(input, output, size);
     
     // Check for kernel launch errors
     CUDA_CHECK(cudaPeekAtLastError());
     CUDA_CHECK(cudaDeviceSynchronize());
 }
 
 /**
  * @brief Host wrapper for Softmax activation function
  * 
  * @param input Pointer to input array in device memory
  * @param output Pointer to output array in device memory
  * @param batch_size Number of vectors (batches)
  * @param dim Dimension of each vector
  */
 void softmax(const float* input, float* output, const int batch_size, const int dim) {
     // For Softmax, one thread block processes an entire batch
     int blockSize = std::min(256, dim);  // Limit to vector dimension or 256
     int numBlocks = batch_size;
     
     // Calculate shared memory size needed for reductions
     int sharedMemSize = blockSize * sizeof(float);
     
     // Launch kernel with shared memory
     softmaxKernel<<<numBlocks, blockSize, sharedMemSize>>>(input, output, batch_size, dim);
     
     // Check for kernel launch errors
     CUDA_CHECK(cudaPeekAtLastError());
     CUDA_CHECK(cudaDeviceSynchronize());
 }
 
 /**
  * @brief Enumeration of activation function types
  */
 enum class ActivationType {
     RELU,
     LEAKY_RELU,
     ELU,
     GELU,
     SOFTMAX
 };
 
 /**
  * @brief Function to benchmark activation functions
  * 
  * @param type Activation function type
  * @param d_input Pointer to input array in device memory
  * @param d_output Pointer to output array in device memory
  * @param size Total number of elements
  * @param batch_size Number of batches (for Softmax)
  * @param dim Dimension of each vector (for Softmax)
  * @param alpha Alpha parameter for LeakyReLU and ELU
  * @return float Execution time in milliseconds
  */
 float benchmarkActivation(ActivationType type, const float* d_input, float* d_output, 
                          int size, int batch_size = 1, int dim = 0, 
                          float alpha = 0.01f) {
     // Create CUDA events for timing
     cudaEvent_t start, stop;
     CUDA_CHECK(cudaEventCreate(&start));
     CUDA_CHECK(cudaEventCreate(&stop));
     
     // Start timing
     CUDA_CHECK(cudaEventRecord(start));
     
     // Call appropriate activation function
     switch (type) {
         case ActivationType::RELU:
             relu(d_input, d_output, size);
             break;
         case ActivationType::LEAKY_RELU:
             leakyRelu(d_input, d_output, size, alpha);
             break;
         case ActivationType::ELU:
             elu(d_input, d_output, size, alpha);
             break;
         case ActivationType::GELU:
             gelu(d_input, d_output, size);
             break;
         case ActivationType::SOFTMAX:
             if (batch_size <= 0 || dim <= 0) {
                 throw std::invalid_argument("Softmax requires valid batch_size and dim parameters");
             }
             softmax(d_input, d_output, batch_size, dim);
             break;
         default:
             throw std::invalid_argument("Unknown activation type");
     }
     
     // Stop timing
     CUDA_CHECK(cudaEventRecord(stop));
     CUDA_CHECK(cudaEventSynchronize(stop));
     
     // Calculate elapsed time
     float milliseconds = 0;
     CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
     
     // Clean up events
     CUDA_CHECK(cudaEventDestroy(start));
     CUDA_CHECK(cudaEventDestroy(stop));
     
     return milliseconds;
 }
 
 /**
  * @brief Function to validate activation function results
  * 
  * @param type Activation function type
  * @param input Vector of input values
  * @param output Vector of output values
  * @param alpha Alpha parameter for LeakyReLU and ELU
  * @param tolerance Error tolerance for floating-point comparisons
  * @return true If validation passes
  * @return false If validation fails
  */
 bool validateActivation(ActivationType type, const std::vector<float>& input, 
                         const std::vector<float>& output, float alpha = 0.01f, 
                         float tolerance = 1e-5f) {
     // Check for size mismatch
     if (input.size() != output.size()) {
         std::cerr << "Size mismatch in validation: input size = " << input.size() 
                   << ", output size = " << output.size() << std::endl;
         return false;
     }
     
     switch (type) {
         case ActivationType::RELU: {
             for (size_t i = 0; i < input.size(); ++i) {
                 float expected = std::max(0.0f, input[i]);
                 if (std::abs(output[i] - expected) > tolerance) {
                     std::cout << "ReLU validation failed at index " << i 
                               << ": expected " << expected 
                               << ", got " << output[i] << std::endl;
                     return false;
                 }
             }
             return true;
         }
         
         case ActivationType::LEAKY_RELU: {
             for (size_t i = 0; i < input.size(); ++i) {
                 float expected = input[i] > 0.0f ? input[i] : alpha * input[i];
                 if (std::abs(output[i] - expected) > tolerance) {
                     std::cout << "LeakyReLU validation failed at index " << i 
                               << ": expected " << expected 
                               << ", got " << output[i] << std::endl;
                     return false;
                 }
             }
             return true;
         }
         
         case ActivationType::ELU: {
             for (size_t i = 0; i < input.size(); ++i) {
                 float expected = input[i] > 0.0f ? input[i] : alpha * (std::exp(input[i]) - 1.0f);
                 if (std::abs(output[i] - expected) > tolerance) {
                     std::cout << "ELU validation failed at index " << i 
                               << ": expected " << expected 
                               << ", got " << output[i] << std::endl;
                     return false;
                 }
             }
             return true;
         }
         
         case ActivationType::GELU: {
             for (size_t i = 0; i < input.size(); ++i) {
                 float x = input[i];
                 float sqrt_2_over_pi = 0.7978845608028654f;
                 float coeff = 0.044715f;
                 float expected = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
                 if (std::abs(output[i] - expected) > tolerance) {
                     std::cout << "GELU validation failed at index " << i 
                               << ": expected " << expected 
                               << ", got " << output[i] << std::endl;
                     return false;
                 }
             }
             return true;
         }
         
         case ActivationType::SOFTMAX: {
             // For Softmax, we need to verify batch by batch
             size_t batch_size = 1;  // Default, can be adjusted as needed
             size_t dim = input.size() / batch_size;
             
             for (size_t b = 0; b < batch_size; ++b) {
                 // Calculate expected softmax values
                 std::vector<float> expected(dim);
                 float max_val = -FLT_MAX;
                 
                 // Find max value for numerical stability
                 for (size_t i = 0; i < dim; ++i) {
                     max_val = std::max(max_val, input[b * dim + i]);
                 }
                 
                 // Compute exp(x - max_val) / sum(exp(x - max_val))
                 float sum = 0.0f;
                 for (size_t i = 0; i < dim; ++i) {
                     expected[i] = std::exp(input[b * dim + i] - max_val);
                     sum += expected[i];
                 }
                 
                 for (size_t i = 0; i < dim; ++i) {
                     expected[i] /= sum;
                     
                     if (std::abs(output[b * dim + i] - expected[i]) > tolerance) {
                         std::cout << "Softmax validation failed at batch " << b 
                                   << ", index " << i 
                                   << ": expected " << expected[i] 
                                   << ", got " << output[b * dim + i] << std::endl;
                         return false;
                     }
                 }
             }
             return true;
         }
         
         default:
             throw std::invalid_argument("Unknown activation type for validation");
     }
 }
 
 /**
  * @brief Main function to demonstrate and benchmark activation functions
  */
 int main() {
     try {
         // Set device properties
         int deviceId = 0;
         cudaDeviceProp deviceProp;
         CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));
         
         std::cout << "Using CUDA device: " << deviceProp.name << std::endl;
         
         // Test parameters
         const int size = 1 << 24;  // ~16M elements
         const int batch_size = 128;
         const int softmax_dim = 1024;
         const float alpha_leaky = 0.01f;
         const float alpha_elu = 1.0f;
         
         std::cout << "Initializing data with " << size << " elements..." << std::endl;
         
         // Allocate host memory
         std::vector<float> h_input(size);
         std::vector<float> h_output(size);
         
         // Initialize input with random values in range [-5.0, 5.0]
         std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
         
         for (int i = 0; i < size; ++i) {
             h_input[i] = dist(gen);
         }
         
         // Allocate device memory
         float *d_input, *d_output;
         CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
         
         // Copy input data to device
         CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));
         
         std::cout << "Starting benchmarks..." << std::endl;
         
         // Benchmark and validate each activation function
         
         // 1. ReLU
         float relu_time = benchmarkActivation(ActivationType::RELU, d_input, d_output, size);
         CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
         bool relu_valid = validateActivation(ActivationType::RELU, h_input, h_output);
         
         // 2. Leaky ReLU
         float leaky_relu_time = benchmarkActivation(ActivationType::LEAKY_RELU, d_input, d_output, size, 1, 0, alpha_leaky);
         CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
         bool leaky_relu_valid = validateActivation(ActivationType::LEAKY_RELU, h_input, h_output, alpha_leaky);
         
         // 3. ELU
         float elu_time = benchmarkActivation(ActivationType::ELU, d_input, d_output, size, 1, 0, alpha_elu);
         CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
         bool elu_valid = validateActivation(ActivationType::ELU, h_input, h_output, alpha_elu);
         
         // 4. GELU
         float gelu_time = benchmarkActivation(ActivationType::GELU, d_input, d_output, size);
         CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
         bool gelu_valid = validateActivation(ActivationType::GELU, h_input, h_output);
         
         // 5. Softmax (using a subset of the data due to batching requirements)
         const int softmax_size = batch_size * softmax_dim;
         
         // Benchmark Softmax with a subset of the data
         float softmax_time = benchmarkActivation(ActivationType::SOFTMAX, d_input, d_output, softmax_size, batch_size, softmax_dim);
         
         // Create a smaller vector for softmax validation
         std::vector<float> h_softmax_input(softmax_size);
         std::vector<float> h_softmax_output(softmax_size);
         
         // Copy subset of input data for softmax
         std::copy(h_input.begin(), h_input.begin() + softmax_size, h_softmax_input.begin());
         
         // Copy softmax results back to host
         CUDA_CHECK(cudaMemcpy(h_softmax_output.data(), d_output, softmax_size * sizeof(float), cudaMemcpyDeviceToHost));
         
         // Validate softmax results
         bool softmax_valid = true;
         for (int b = 0; b < batch_size && softmax_valid; ++b) {
             // Calculate expected softmax values
             std::vector<float> expected(softmax_dim);
             float max_val = -FLT_MAX;
             
             // Find max value
             for (int i = 0; i < softmax_dim; ++i) {
                 max_val = std::max(max_val, h_softmax_input[b * softmax_dim + i]);
             }
             
             // Compute softmax
             float sum = 0.0f;
             for (int i = 0; i < softmax_dim; ++i) {
                 expected[i] = std::exp(h_softmax_input[b * softmax_dim + i] - max_val);
                 sum += expected[i];
             }
             
             for (int i = 0; i < softmax_dim; ++i) {
                 expected[i] /= sum;
                 if (std::abs(h_softmax_output[b * softmax_dim + i] - expected[i]) > 1e-5f) {
                     softmax_valid = false;
                     std::cout << "Softmax validation failed at batch " << b 
                               << ", index " << i 
                               << ": expected " << expected[i] 
                               << ", got " << h_softmax_output[b * softmax_dim + i] << std::endl;
                     break;
                 }
             }
         }
         
         // Print benchmark results
         std::cout << "\n===== Activation Functions Benchmark Results =====\n";
         std::cout << "Data size: " << size << " elements\n";
         std::cout << "Softmax: " << batch_size << " batches with " << softmax_dim << " elements each\n\n";
         
         std::cout << "ReLU:       " << relu_time << " ms, Valid: " << (relu_valid ? "Yes" : "No") << "\n";
         std::cout << "Leaky ReLU: " << leaky_relu_time << " ms, Valid: " << (leaky_relu_valid ? "Yes" : "No") << "\n";
         std::cout << "ELU:        " << elu_time << " ms, Valid: " << (elu_valid ? "Yes" : "No") << "\n";
         std::cout << "GELU:       " << gelu_time << " ms, Valid: " << (gelu_valid ? "Yes" : "No") << "\n";
         std::cout << "Softmax:    " << softmax_time << " ms, Valid: " << (softmax_valid ? "Yes" : "No") << "\n";
         
         // Perform simple tests with known inputs for visualization
         const int test_size = 5;
         std::vector<float> test_input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
         std::vector<float> test_output(test_size);
         
         // Copy test data to device
         CUDA_CHECK(cudaMemcpy(d_input, test_input.data(), test_size * sizeof(float), cudaMemcpyHostToDevice));
         
         // Test ReLU
         relu(d_input, d_output, test_size);
         CUDA_CHECK(cudaMemcpy(test_output.data(), d_output, test_size * sizeof(float), cudaMemcpyDeviceToHost));
         
         std::cout << "\n===== Activation Function Test Results =====\n";
         std::cout << "ReLU Test:\n";
         std::cout << "Input:  ";
         for (int i = 0; i < test_size; ++i) std::cout << test_input[i] << " ";
         std::cout << "\nOutput: ";
         for (int i = 0; i < test_size; ++i) std::cout << test_output[i] << " ";
         std::cout << "\n";
         
         // Test Leaky ReLU
         leakyRelu(d_input, d_output, test_size, alpha_leaky);
         CUDA_CHECK(cudaMemcpy(test_output.data(), d_output, test_size * sizeof(float), cudaMemcpyDeviceToHost));
         
         std::cout << "Leaky ReLU Test (alpha=" << alpha_leaky << "):\n";
         std::cout << "Input:  ";
         for (int i = 0; i < test_size; ++i) std::cout << test_input[i] << " ";
         std::cout << "\nOutput: ";
         for (int i = 0; i < test_size; ++i) std::cout << test_output[i] << " ";
         std::cout << "\n";
         
         // Test ELU
         elu(d_input, d_output, test_size, alpha_elu);
         CUDA_CHECK(cudaMemcpy(test_output.data(), d_output, test_size * sizeof(float), cudaMemcpyDeviceToHost));
         
         std::cout << "ELU Test (alpha=" << alpha_elu << "):\n";
         std::cout << "Input:  ";
         for (int i = 0; i < test_size; ++i) std::cout << test_input[i] << " ";
         std::cout << "\nOutput: ";
         for (int i = 0; i < test_size; ++i) std::cout << test_output[i] << " ";
         std::cout << "\n";
         
         // Test GELU
         gelu(d_input, d_output, test_size);
         CUDA_CHECK(cudaMemcpy(test_output.data(), d_output, test_size * sizeof(float), cudaMemcpyDeviceToHost));
         
         std::cout << "GELU Test:\n";
         std::cout << "Input:  ";
         for (int i = 0; i < test_size; ++i) std::cout << test_input[i] << " ";
         std::cout << "\nOutput: ";
         for (int i = 0; i < test_size; ++i) std::cout << test_output[i] << " ";
         std::cout << "\n";
         
         // Test Softmax (treating the small array as a single batch)
         softmax(d_input, d_output, 1, test_size);
         CUDA_CHECK(cudaMemcpy(test_output.data(), d_output, test_size * sizeof(float), cudaMemcpyDeviceToHost));
         
         std::cout << "Softmax Test:\n";
         std::cout << "Input:  ";
         for (int i = 0; i < test_size; ++i) std::cout << test_input[i] << " ";
         std::cout << "\nOutput: ";
         for (int i = 0; i < test_size; ++i) std::cout << test_output[i] << " ";
         std::cout << "\n";
         
         // Verify sum of softmax outputs equals 1.0
         float softmax_sum = 0.0f;
         for (int i = 0; i < test_size; ++i) {
             softmax_sum += test_output[i];
         }
         std::cout << "Softmax sum = " << softmax_sum << " (should be close to 1.0)\n";
         
         // Free device memory
         CUDA_CHECK(cudaFree(d_input));
         CUDA_CHECK(cudaFree(d_output));
         
         return EXIT_SUCCESS;
     }
     catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return EXIT_FAILURE;
     }
 }