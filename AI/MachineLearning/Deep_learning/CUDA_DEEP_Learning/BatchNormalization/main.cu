/**
 * @file batch_normalization.cu
 * @brief GPU-based batch normalization layer implementation using CUDA
 * 
 * This implementation provides a complete batch normalization layer for deep neural networks
 * with both forward and backward passes. Batch normalization helps stabilize training by
 * normalizing the activations, which allows for higher learning rates and faster convergence.
 * 
 * Time Complexity:
 * - Forward pass: O(N), where N is the number of elements (batch_size * feature_size * spatial_size)
 * - Backward pass: O(N), where N is the number of elements (batch_size * feature_size * spatial_size)
 * 
 * Space Complexity:
 * - O(N + F), where N is the number of elements and F is the number of features
 */

 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <stdexcept>
 #include <mutex>
 
 // Error checking macro for CUDA calls to improve debugging
 #define CUDA_CHECK(call) \
     do { \
         cudaError_t error = call; \
         if (error != cudaSuccess) { \
             std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
             throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
         } \
     } while(0)
 
 // Small epsilon value to avoid division by zero in normalization
 constexpr float EPSILON = 1e-5f;
 
 //------------------------------------------------------------------------------
 // CUDA kernel declarations (must be outside class)
 //------------------------------------------------------------------------------
 
 /**
  * @brief CUDA kernel to compute mean across batch for each feature
  */
 __global__ void computeMeanKernel(const float* input, float* mean,
                                  int batch_size, int feature_size, int spatial_size) {
     // Each thread processes one feature
     int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (feature_idx < feature_size) {
         float sum = 0.0f;
         int count = batch_size * spatial_size;
         
         // Sum all values for this feature across the batch and spatial dimensions
         for (int n = 0; n < batch_size; ++n) {
             for (int s = 0; s < spatial_size; ++s) {
                 int idx = n * feature_size * spatial_size + feature_idx * spatial_size + s;
                 sum += input[idx];
             }
         }
         
         // Calculate mean
         mean[feature_idx] = sum / count;
     }
 }
 
 /**
  * @brief CUDA kernel to compute variance across batch for each feature
  */
 __global__ void computeVarianceKernel(const float* input, const float* mean, float* var,
                                      int batch_size, int feature_size, int spatial_size) {
     // Each thread processes one feature
     int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (feature_idx < feature_size) {
         float sum_squared_diff = 0.0f;
         int count = batch_size * spatial_size;
         float feature_mean = mean[feature_idx];
         
         // Sum squared differences from mean for this feature
         for (int n = 0; n < batch_size; ++n) {
             for (int s = 0; s < spatial_size; ++s) {
                 int idx = n * feature_size * spatial_size + feature_idx * spatial_size + s;
                 float diff = input[idx] - feature_mean;
                 sum_squared_diff += diff * diff;
             }
         }
         
         // Calculate variance
         var[feature_idx] = sum_squared_diff / count;
     }
 }
 
 /**
  * @brief CUDA kernel to update running statistics
  */
 __global__ void updateRunningStatsKernel(const float* batch_mean, const float* batch_var,
                                         float* running_mean, float* running_var,
                                         float momentum, int feature_size) {
     // Each thread processes one feature
     int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (feature_idx < feature_size) {
         // Update running mean: running_mean = momentum * running_mean + (1 - momentum) * batch_mean
         running_mean[feature_idx] = momentum * running_mean[feature_idx] + 
                                    (1.0f - momentum) * batch_mean[feature_idx];
         
         // Update running variance: running_var = momentum * running_var + (1 - momentum) * batch_var
         running_var[feature_idx] = momentum * running_var[feature_idx] + 
                                   (1.0f - momentum) * batch_var[feature_idx];
     }
 }
 
 /**
  * @brief CUDA kernel for forward pass of batch normalization
  */
 __global__ void batchNormForwardKernel(const float* input, float* output,
                                       const float* mean, const float* var,
                                       const float* gamma, const float* beta,
                                       float epsilon, int batch_size, int feature_size, int spatial_size) {
     // Calculate global thread index
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     int total_elements = batch_size * feature_size * spatial_size;
     
     if (idx < total_elements) {
         // Calculate indices
         int n = idx / (feature_size * spatial_size);                        // Batch index
         int f = (idx % (feature_size * spatial_size)) / spatial_size;       // Feature index
         int s = idx % spatial_size;                                         // Spatial index
         
         // Get mean and variance for this feature
         float feature_mean = mean[f];
         float feature_var = var[f];
         float feature_gamma = gamma[f];
         float feature_beta = beta[f];
         
         // Normalize: (x - mean) / sqrt(var + epsilon)
         float normalized = (input[idx] - feature_mean) / sqrtf(feature_var + epsilon);
         
         // Scale and shift: gamma * normalized + beta
         output[idx] = feature_gamma * normalized + feature_beta;
     }
 }
 
 /**
  * @brief CUDA kernel to compute gradients for gamma and beta
  */
 __global__ void computeParamGradientsKernel(const float* input, const float* grad_output,
                                            const float* mean, const float* var,
                                            float* grad_gamma, float* grad_beta,
                                            float epsilon, int batch_size, int feature_size, int spatial_size) {
     // Each thread processes one feature
     int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (feature_idx < feature_size) {
         float sum_dy = 0.0f;
         float sum_dy_xhat = 0.0f;
         float feature_mean = mean[feature_idx];
         float feature_var = var[feature_idx];
         float inv_std = 1.0f / sqrtf(feature_var + epsilon);
         
         // Sum gradients for this feature
         for (int n = 0; n < batch_size; ++n) {
             for (int s = 0; s < spatial_size; ++s) {
                 int idx = n * feature_size * spatial_size + feature_idx * spatial_size + s;
                 float xhat = (input[idx] - feature_mean) * inv_std;
                 sum_dy += grad_output[idx];
                 sum_dy_xhat += grad_output[idx] * xhat;
             }
         }
         
         // Calculate gradients for gamma and beta
         grad_beta[feature_idx] = sum_dy;
         grad_gamma[feature_idx] = sum_dy_xhat;
     }
 }
 
 /**
  * @brief CUDA kernel to compute gradient with respect to input
  */
 __global__ void computeInputGradientKernel(const float* input, const float* grad_output,
                                           const float* mean, const float* var,
                                           const float* gamma, float* grad_input,
                                           float epsilon, int batch_size, int feature_size, int spatial_size) {
     // Calculate global thread index
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     int total_elements = batch_size * feature_size * spatial_size;
     
     if (idx < total_elements) {
         // Calculate indices
         int n = idx / (feature_size * spatial_size);                        // Batch index
         int f = (idx % (feature_size * spatial_size)) / spatial_size;       // Feature index
         int s = idx % spatial_size;                                         // Spatial index
         
         // Get values for this feature
         float feature_mean = mean[f];
         float feature_var = var[f];
         float feature_gamma = gamma[f];
         float inv_std = 1.0f / sqrtf(feature_var + epsilon);
         int count = batch_size * spatial_size;
         
         // Calculate gradients for this element
         float x_centered = input[idx] - feature_mean;
         float dy = grad_output[idx];
         
         // Intermediate values for gradient calculation
         float sum_dy = 0.0f;
         float sum_dy_xhat = 0.0f;
         
         // Sum gradients across batch for this feature
         for (int n_inner = 0; n_inner < batch_size; ++n_inner) {
             for (int s_inner = 0; s_inner < spatial_size; ++s_inner) {
                 int inner_idx = n_inner * feature_size * spatial_size + f * spatial_size + s_inner;
                 float inner_x_centered = input[inner_idx] - feature_mean;
                 sum_dy += grad_output[inner_idx];
                 sum_dy_xhat += grad_output[inner_idx] * inner_x_centered * inv_std;
             }
         }
         
         // Calculate gradient for input using chain rule
         // dx = (1/N) * gamma * (var + epsilon)^(-1/2) * (N * dy - sum(dy) - (x - mean) * (var + epsilon)^(-1) * sum(dy * (x - mean)))
         grad_input[idx] = feature_gamma * inv_std * (dy - sum_dy / count - x_centered * inv_std * inv_std * sum_dy_xhat / count);
     }
 }
 
 /**
  * @class BatchNormalization
  * @brief Implements a GPU-based batch normalization layer for deep neural networks
  * 
  * This class provides functionality for normalizing activations across the batch dimension,
  * which helps with training stability and faster convergence.
  */
 class BatchNormalization {
 public:
     /**
      * @brief Constructor for BatchNormalization
      * 
      * @param feature_size Number of features (channels) to normalize
      * @param momentum Momentum factor for running statistics (default: 0.9)
      * @param epsilon Small constant to avoid division by zero (default: 1e-5)
      */
     BatchNormalization(int feature_size, float momentum = 0.9f, float epsilon = EPSILON) 
         : feature_size_(feature_size), momentum_(momentum), epsilon_(epsilon), 
           is_training_(true) {
         
         // Allocate memory for learnable parameters (gamma and beta)
         CUDA_CHECK(cudaMalloc(&g   , feature_size_ * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&beta_, feature_size_ * sizeof(float)));
         
         // Allocate memory for running mean and variance (used during inference)
         CUDA_CHECK(cudaMalloc(&running_mean_, feature_size_ * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&running_var_, feature_size_ * sizeof(float)));
         
         // Allocate memory for batch mean and variance (used during training)
         CUDA_CHECK(cudaMalloc(&batch_mean_, feature_size_ * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&batch_var_, feature_size_ * sizeof(float)));
         
         // Initialize gamma to 1 and beta to 0 (default values)
         std::vector<float> h_gamma(feature_size_, 1.0f);
         std::vector<float> h_beta(feature_size_, 0.0f);
         CUDA_CHECK(cudaMemcpy(gamma_, h_gamma.data(), feature_size_ * sizeof(float), cudaMemcpyHostToDevice));
         CUDA_CHECK(cudaMemcpy(beta_, h_beta.data(), feature_size_ * sizeof(float), cudaMemcpyHostToDevice));
         
         // Initialize running mean and variance to 0
         std::vector<float> h_zeros(feature_size_, 0.0f);
         CUDA_CHECK(cudaMemcpy(running_mean_, h_zeros.data(), feature_size_ * sizeof(float), cudaMemcpyHostToDevice));
         CUDA_CHECK(cudaMemcpy(running_var_, h_zeros.data(), feature_size_ * sizeof(float), cudaMemcpyHostToDevice));
         
         // Create CUDA stream for asynchronous execution
         CUDA_CHECK(cudaStreamCreate(&stream_));
     }
     
     /**
      * @brief Destructor - frees allocated CUDA memory
      */
     ~BatchNormalization() {
         // Free allocated CUDA memory
         CUDA_CHECK(cudaFree(gamma_));
         CUDA_CHECK(cudaFree(beta_));
         CUDA_CHECK(cudaFree(running_mean_));
         CUDA_CHECK(cudaFree(running_var_));
         CUDA_CHECK(cudaFree(batch_mean_));
         CUDA_CHECK(cudaFree(batch_var_));
         
         // Destroy CUDA stream
         CUDA_CHECK(cudaStreamDestroy(stream_));
     }
     
     /**
      * @brief Set the training mode
      * 
      * @param is_training If true, use batch statistics; if false, use running statistics
      */
     void setTrainingMode(bool is_training) {
         // Thread-safe mode change using mutex lock
         std::lock_guard<std::mutex> lock(mutex_);
         is_training_ = is_training;
     }
     
     /**
      * @brief Forward pass of batch normalization
      * 
      * @param input Input tensor (device pointer)
      * @param output Output tensor (device pointer)
      * @param batch_size Number of samples in the batch
      * @param spatial_size Spatial size of each feature (e.g., height * width for 2D data)
      * @return void
      */
     void forward(const float* input, float* output, int batch_size, int spatial_size) {
         // Thread-safe forward pass using mutex lock
         std::lock_guard<std::mutex> lock(mutex_);
         
         if (is_training_) {
             // In training mode: compute batch statistics and update running statistics
             
             // Compute mean and variance for this batch
             computeBatchStatistics(input, batch_size, spatial_size);
             
             // Update running statistics
             updateRunningStatistics();
             
             // Normalize using batch statistics
             normalize(input, output, batch_mean_, batch_var_, batch_size, spatial_size);
         } else {
             // In inference mode: use running statistics
             normalize(input, output, running_mean_, running_var_, batch_size, spatial_size);
         }
     }
     
     /**
      * @brief Backward pass of batch normalization
      * 
      * @param input Original input from forward pass (device pointer)
      * @param grad_output Gradient of loss with respect to output (device pointer)
      * @param grad_input Gradient of loss with respect to input (device pointer)
      * @param grad_gamma Gradient of loss with respect to gamma (device pointer)
      * @param grad_beta Gradient of loss with respect to beta (device pointer)
      * @param batch_size Number of samples in the batch
      * @param spatial_size Spatial size of each feature
      * @return void
      */
     void backward(const float* input, const float* grad_output, 
                   float* grad_input, float* grad_gamma, float* grad_beta,
                   int batch_size, int spatial_size) {
         // Thread-safe backward pass using mutex lock
         std::lock_guard<std::mutex> lock(mutex_);
         
         // Only perform backward pass if in training mode
         if (!is_training_) {
             throw std::runtime_error("Backward pass called while not in training mode");
         }
         
         // Calculate gradients for learnable parameters and input
         batchNormBackward(input, grad_output, grad_input, grad_gamma, grad_beta,
                           batch_size, spatial_size);
     }
     
     /**
      * @brief Get the learnable parameter gamma
      * 
      * @param h_gamma Host pointer to store gamma values
      * @return void
      */
     void getGamma(float* h_gamma) const {
         // Thread-safe memory operation using mutex lock
         std::lock_guard<std::mutex> lock(mutex_);
         CUDA_CHECK(cudaMemcpy(h_gamma, gamma_, feature_size_ * sizeof(float), cudaMemcpyDeviceToHost));
     }
     
     /**
      * @brief Get the learnable parameter beta
      * 
      * @param h_beta Host pointer to store beta values
      * @return void
      */
     void getBeta(float* h_beta) const {
         // Thread-safe memory operation using mutex lock
         std::lock_guard<std::mutex> lock(mutex_);
         CUDA_CHECK(cudaMemcpy(h_beta, beta_, feature_size_ * sizeof(float), cudaMemcpyDeviceToHost));
     }
     
     /**
      * @brief Set the learnable parameter gamma
      * 
      * @param h_gamma Host pointer containing gamma values
      * @return void
      */
     void setGamma(const float* h_gamma) {
         // Thread-safe memory operation using mutex lock
         std::lock_guard<std::mutex> lock(mutex_);
         CUDA_CHECK(cudaMemcpy(gamma_, h_gamma, feature_size_ * sizeof(float), cudaMemcpyHostToDevice));
     }
     
     /**
      * @brief Set the learnable parameter beta
      * 
      * @param h_beta Host pointer containing beta values
      * @return void
      */
     void setBeta(const float* h_beta) {
         // Thread-safe memory operation using mutex lock
         std::lock_guard<std::mutex> lock(mutex_);
         CUDA_CHECK(cudaMemcpy(beta_, h_beta, feature_size_ * sizeof(float), cudaMemcpyHostToDevice));
     }
 
 private:
     // Device pointers for learnable parameters
     float* gamma_;       // Scale parameter
     float* beta_;        // Shift parameter
     
     // Device pointers for statistics
     float* running_mean_;  // Running mean (for inference)
     float* running_var_;   // Running variance (for inference)
     float* batch_mean_;    // Batch mean (for training)
     float* batch_var_;     // Batch variance (for training)
     
     // Configuration parameters
     int feature_size_;     // Number of features (channels)
     float momentum_;       // Momentum for running statistics update
     float epsilon_;        // Small constant for numerical stability
     bool is_training_;     // Training mode flag
     
     // CUDA stream for asynchronous execution
     cudaStream_t stream_;
     
     // Mutex for thread safety
     // Lock is needed when multiple threads might access the batch normalization layer concurrently
     mutable std::mutex mutex_;
 
     /**
      * @brief Compute mean and variance for the current batch
      * 
      * @param input Input tensor (device pointer)
      * @param batch_size Number of samples in the batch
      * @param spatial_size Spatial size of each feature
      * @return void
      */
     void computeBatchStatistics(const float* input, int batch_size, int spatial_size) {
         // Determine grid and block dimensions
         int threads_per_block = 256;
         int blocks = (feature_size_ + threads_per_block - 1) / threads_per_block;
         
         // Launch kernel to compute mean
         computeMeanKernel<<<blocks, threads_per_block, 0, stream_>>>(
             input, batch_mean_, batch_size, feature_size_, spatial_size);
         
         // Check for errors
         CUDA_CHECK(cudaGetLastError());
         
         // Launch kernel to compute variance
         computeVarianceKernel<<<blocks, threads_per_block, 0, stream_>>>(
             input, batch_mean_, batch_var_, batch_size, feature_size_, spatial_size);
         
         // Check for errors
         CUDA_CHECK(cudaGetLastError());
     }
     
     /**
      * @brief Update running statistics using batch statistics
      * 
      * @return void
      */
     void updateRunningStatistics() {
         // Determine grid and block dimensions
         int threads_per_block = 256;
         int blocks = (feature_size_ + threads_per_block - 1) / threads_per_block;
         
         // Launch kernel to update running statistics
         updateRunningStatsKernel<<<blocks, threads_per_block, 0, stream_>>>(
             batch_mean_, batch_var_, running_mean_, running_var_, momentum_, feature_size_);
         
         // Check for errors
         CUDA_CHECK(cudaGetLastError());
     }
     
     /**
      * @brief Normalize input using given mean and variance
      * 
      * @param input Input tensor (device pointer)
      * @param output Output tensor (device pointer)
      * @param mean Mean values (device pointer)
      * @param var Variance values (device pointer)
      * @param batch_size Number of samples in the batch
      * @param spatial_size Spatial size of each feature
      * @return void
      */
     void normalize(const float* input, float* output, const float* mean, const float* var,
                   int batch_size, int spatial_size) {
         // Total number of elements
         int num_elements = batch_size * feature_size_ * spatial_size;
         
         // Determine grid and block dimensions
         int threads_per_block = 256;
         int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
         
         // Launch kernel
         batchNormForwardKernel<<<blocks, threads_per_block, 0, stream_>>>(
             input, output, mean, var, gamma_, beta_, epsilon_, batch_size, feature_size_, spatial_size);
         
         // Check for errors
         CUDA_CHECK(cudaGetLastError());
     }
     
     /**
      * @brief Function to perform backward pass of batch normalization
      */
     void batchNormBackward(const float* input, const float* grad_output,
                           float* grad_input, float* grad_gamma, float* grad_beta,
                           int batch_size, int spatial_size) {
         // Total number of elements
         int num_elements = batch_size * feature_size_ * spatial_size;
         
         // Determine grid and block dimensions for param gradients
         int threads_per_block_params = 256;
         int blocks_params = (feature_size_ + threads_per_block_params - 1) / threads_per_block_params;
         
         // Launch kernel to compute gradients for gamma and beta
         computeParamGradientsKernel<<<blocks_params, threads_per_block_params, 0, stream_>>>(
             input, grad_output, batch_mean_, batch_var_,
             grad_gamma, grad_beta, epsilon_, batch_size, feature_size_, spatial_size);
         
         // Check for errors
         CUDA_CHECK(cudaGetLastError());
         
         // Determine grid and block dimensions for input gradients
         int threads_per_block_input = 256;
         int blocks_input = (num_elements + threads_per_block_input - 1) / threads_per_block_input;
         
         // Launch kernel to compute gradient with respect to input
         computeInputGradientKernel<<<blocks_input, threads_per_block_input, 0, stream_>>>(
             input, grad_output, batch_mean_, batch_var_,
             gamma_, grad_input, epsilon_, batch_size, feature_size_, spatial_size);
         
         // Check for errors
         CUDA_CHECK(cudaGetLastError());
     }
 };
 
 /**
  * @brief Test function to demonstrate batch normalization usage
  */
 void testBatchNormalization() {
     try {
         // Set CUDA device
         int device_id = 0;
         CUDA_CHECK(cudaSetDevice(device_id));
         
         // Parameters for the test
         const int batch_size = 32;
         const int feature_size = 64;
         const int spatial_size = 28 * 28;  // Example: 28x28 feature maps
         const int num_elements = batch_size * feature_size * spatial_size;
         
         // Create batch normalization layer
         BatchNormalization bn_layer(feature_size);
         
         // Allocate host memory
         std::vector<float> h_input(num_elements);
         std::vector<float> h_output(num_elements);
         std::vector<float> h_grad_output(num_elements);
         std::vector<float> h_grad_input(num_elements);
         std::vector<float> h_grad_gamma(feature_size);
         std::vector<float> h_grad_beta(feature_size);
         
         // Initialize input data with random values (0.0-1.0)
         for (int i = 0; i < num_elements; ++i) {
             h_input[i] = static_cast<float>(rand()) / RAND_MAX;
         }
         
         // Initialize gradient of output with random values (-0.5 to 0.5)
         for (int i = 0; i < num_elements; ++i) {
             h_grad_output[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
         }
         
         // Allocate device memory
         float *d_input, *d_output, *d_grad_output, *d_grad_input, *d_grad_gamma, *d_grad_beta;
         CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&d_grad_output, num_elements * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&d_grad_input, num_elements * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&d_grad_gamma, feature_size * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&d_grad_beta, feature_size * sizeof(float)));
         
         // Copy data from host to device
         CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
         CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
         
         // Test training mode forward and backward
         std::cout << "Testing batch normalization in training mode..." << std::endl;
         bn_layer.setTrainingMode(true);
         
         // Forward pass
         bn_layer.forward(d_input, d_output, batch_size, spatial_size);
         
         // Backward pass
         bn_layer.backward(d_input, d_grad_output, d_grad_input, d_grad_gamma, d_grad_beta, batch_size, spatial_size);
         
         // Copy results from device to host
         CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
         CUDA_CHECK(cudaMemcpy(h_grad_input.data(), d_grad_input, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
         CUDA_CHECK(cudaMemcpy(h_grad_gamma.data(), d_grad_gamma, feature_size * sizeof(float), cudaMemcpyDeviceToHost));
         CUDA_CHECK(cudaMemcpy(h_grad_beta.data(), d_grad_beta, feature_size * sizeof(float), cudaMemcpyDeviceToHost));
         
         // Print sample results
         std::cout << "Sample output values (training):" << std::endl;
         for (int i = 0; i < 5; ++i) {
             std::cout << "Output[" << i << "] = " << h_output[i] << std::endl;
         }
         
         std::cout << "\nSample gradient values:" << std::endl;
         for (int i = 0; i < 5; ++i) {
             std::cout << "Grad_input[" << i << "] = " << h_grad_input[i] << std::endl;
         }
         
         // Test inference mode
         std::cout << "\nTesting batch normalization in inference mode..." << std::endl;
         bn_layer.setTrainingMode(false);
         
         // Forward pass in inference mode
         bn_layer.forward(d_input, d_output, batch_size, spatial_size);
         
         // Copy inference results from device to host
         CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost));
         
         // Print sample inference results
         std::cout << "Sample output values (inference):" << std::endl;
         for (int i = 0; i < 5; ++i) {
             std::cout << "Output[" << i << "] = " << h_output[i] << std::endl;
         }
         
         // Free device memory
         CUDA_CHECK(cudaFree(d_input));
         CUDA_CHECK(cudaFree(d_output));
         CUDA_CHECK(cudaFree(d_grad_output));
         CUDA_CHECK(cudaFree(d_grad_input));
         CUDA_CHECK(cudaFree(d_grad_gamma));
         CUDA_CHECK(cudaFree(d_grad_beta));
         
         std::cout << "\nBatch normalization test completed successfully!" << std::endl;
         
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
     }
 }
 
 /**
  * @brief Main function
  */
 int main() {
     try {
         // Test batch normalization
         testBatchNormalization();
         return 0;
     } catch (const std::exception& e) {
         std::cerr << "Fatal error: " << e.what() << std::endl;
         return 1;
     }
 }