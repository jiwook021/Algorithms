/**
 * GPU-accelerated Multi-Layer Perceptron (MLP) for MNIST Digit Classification
 * 
 * This implementation uses CUDA to train and evaluate a simple neural network
 * for classifying handwritten digits from the MNIST dataset.
 * 
 * Network architecture:
 * - Input layer: 784 neurons (28x28 MNIST images)
 * - Hidden layer: 128 neurons with ReLU activation
 * - Output layer: 10 neurons (digits 0-9) with softmax activation
 * 
 * Time complexity:
 * - Forward pass: O(B*I*H + B*H*O) where B=batch size, I=input size, H=hidden size, O=output size
 * - Backward pass: O(B*I*H + B*H*O)
 * - Overall training: O(E*N*(B*I*H + B*H*O)) where E=epochs, N=number of batches
 * 
 * Space complexity:
 * - Model parameters: O(I*H + H*O + H + O)
 * - Batch data: O(B*I + B)
 * - Intermediate activations: O(B*H + B*O)
 * - Total: O(I*H + H*O + B*I + B*H + B*O)
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 #include <cuda_runtime.h>
 #include <time.h>
 
 // Define network architecture
 #define INPUT_SIZE 784   // 28x28 MNIST images
 #define HIDDEN_SIZE 128  // Number of neurons in hidden layer
 #define OUTPUT_SIZE 10   // 10 classes (digits 0-9)
 #define BATCH_SIZE 128   // Number of samples per batch
 #define LEARNING_RATE 0.01f
 #define NUM_EPOCHS 10
 #define MNIST_TRAIN_SIZE 60000
 #define MNIST_TEST_SIZE 10000
 
 /**
  * Error checking macro for CUDA calls
  * Automatically checks for errors after CUDA API calls and exits with error message if detected
  */
 #define CUDA_CHECK(call) \
 do { \
     cudaError_t error = call; \
     if (error != cudaSuccess) { \
         fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                 __FILE__, __LINE__, cudaGetErrorString(error)); \
         exit(EXIT_FAILURE); \
     } \
 } while(0)
 
 // Function declarations
 void loadMNISTData(const char* imageFile, const char* labelFile, 
                   float* images, unsigned char* labels, int numSamples);
 void initializeWeights(float* weights, int inputSize, int outputSize);
 void initializeBiases(float* biases, int size);
 void trainNetwork(float* trainImages, unsigned char* trainLabels, 
                  float* weights1, float* weights2, float* biases1, float* biases2);
 void testNetwork(float* testImages, unsigned char* testLabels, 
                 float* weights1, float* weights2, float* biases1, float* biases2);
 void printLayerInfo(const float* weights, int inputSize, int outputSize, const char* layerName);
 
 /**
  * CUDA kernel for matrix multiplication: C = A * B
  * Each thread computes one element of the output matrix C
  * 
  * @param A - Input matrix A (dimensions: A_rows x A_cols)
  * @param B - Input matrix B (dimensions: A_cols x B_cols)
  * @param C - Output matrix C (dimensions: A_rows x B_cols)
  * @param A_rows - Number of rows in matrix A
  * @param A_cols - Number of columns in matrix A / rows in matrix B
  * @param B_cols - Number of columns in matrix B
  * 
  * Time Complexity: O(A_rows * A_cols * B_cols)
  * Space Complexity: O(A_rows * B_cols)
  */
 __global__ void matrixMultiply(const float* A, const float* B, float* C, 
                               int A_rows, int A_cols, int B_cols) {
     // Calculate global thread indices
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Check if thread is within bounds
     if (row < A_rows && col < B_cols) {
         float sum = 0.0f;
         for (int i = 0; i < A_cols; i++) {
             sum += A[row * A_cols + i] * B[i * B_cols + col];
         }
         C[row * B_cols + col] = sum;
     }
 }
 
 /**
  * CUDA kernel for ReLU activation function: output = max(0, input)
  * 
  * @param input - Input array
  * @param output - Output array (result of ReLU operation)
  * @param size - Size of the input and output arrays
  */
 __global__ void reluActivation(const float* input, float* output, int size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) {
         output[idx] = fmaxf(0.0f, input[idx]);
     }
 }
 
 /**
  * CUDA kernel for ReLU derivative: output = (input > 0) ? 1 : 0
  * Used during backpropagation
  * 
  * @param input - Input array (pre-activation values)
  * @param output - Output array (derivative of ReLU)
  * @param size - Size of the input and output arrays
  */
 __global__ void reluDerivative(const float* input, float* output, int size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) {
         output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
     }
 }
 
 /**
  * CUDA kernel for softmax activation function
  * Computes normalized exponentials for each element in a batch
  * 
  * @param input - Input array (pre-activation values)
  * @param output - Output array (softmax probabilities)
  * @param batch_size - Number of samples in batch
  * @param output_size - Size of output for each sample
  */
 __global__ void softmaxActivation(const float* input, float* output, 
                                  int batch_size, int output_size) {
     int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (batch_idx < batch_size) {
         // Find max for numerical stability
         float max_val = input[batch_idx * output_size];
         for (int i = 1; i < output_size; i++) {
             max_val = fmaxf(max_val, input[batch_idx * output_size + i]);
         }
         
         // Calculate exp(x - max) and sum
         float sum = 0.0f;
         for (int i = 0; i < output_size; i++) {
             output[batch_idx * output_size + i] = expf(input[batch_idx * output_size + i] - max_val);
             sum += output[batch_idx * output_size + i];
         }
         
         // Normalize by sum
         for (int i = 0; i < output_size; i++) {
             output[batch_idx * output_size + i] /= sum;
         }
     }
 }
 
 /**
  * CUDA kernel for cross-entropy loss calculation
  * 
  * @param predictions - Predicted probabilities (after softmax)
  * @param labels - Ground truth labels
  * @param loss - Scalar output for loss (will be atomically updated)
  * @param batch_size - Number of samples in batch
  * @param num_classes - Number of output classes
  */
 __global__ void crossEntropyLoss(const float* predictions, const unsigned char* labels, 
                                 float* loss, int batch_size, int num_classes) {
     int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (batch_idx < batch_size) {
         int label = labels[batch_idx];
         // Ensure valid label (defensive programming)
         if (label >= 0 && label < num_classes) {
             // Clip prediction to avoid log(0)
             float predicted_prob = fmaxf(predictions[batch_idx * num_classes + label], 1e-15f);
             atomicAdd(loss, -logf(predicted_prob));
         }
     }
 }
 
 /**
  * CUDA kernel for calculating output layer error gradient
  * 
  * @param predictions - Predicted probabilities (after softmax)
  * @param labels - Ground truth labels
  * @param error - Output gradient (dL/dz for output layer)
  * @param batch_size - Number of samples in batch
  * @param num_classes - Number of output classes
  */
 __global__ void outputLayerError(const float* predictions, const unsigned char* labels, 
                                float* error, int batch_size, int num_classes) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     int batch_idx = idx / num_classes;
     int class_idx = idx % num_classes;
     
     if (batch_idx < batch_size && class_idx < num_classes) {
         // Calculate softmax gradient (one-hot - predicted)
         if (labels[batch_idx] == class_idx) {
             error[idx] = predictions[idx] - 1.0f;
         } else {
             error[idx] = predictions[idx];
         }
     }
 }
 
 /**
  * CUDA kernel for calculating hidden layer error gradient
  * 
  * @param output_error - Error gradient from output layer
  * @param weights - Weights connecting hidden to output layer
  * @param relu_derivative - Derivative of ReLU activation
  * @param hidden_error - Output gradient for hidden layer
  * @param batch_size - Number of samples in batch
  * @param hidden_size - Size of hidden layer
  * @param output_size - Size of output layer
  */
 __global__ void hiddenLayerError(const float* output_error, const float* weights, 
                                const float* relu_derivative, float* hidden_error,
                                int batch_size, int hidden_size, int output_size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     int batch_idx = idx / hidden_size;
     int hidden_idx = idx % hidden_size;
     
     if (batch_idx < batch_size && hidden_idx < hidden_size) {
         float error_sum = 0.0f;
         for (int i = 0; i < output_size; i++) {
             error_sum += output_error[batch_idx * output_size + i] * 
                          weights[hidden_idx * output_size + i];
         }
         hidden_error[idx] = error_sum * relu_derivative[idx];
     }
 }
 
 /**
  * CUDA kernel for updating weights based on gradients
  * 
  * @param weights - Weight matrix to update
  * @param input - Input activations
  * @param error - Error gradient
  * @param learning_rate - Learning rate for gradient descent
  * @param input_size - Number of input neurons
  * @param output_size - Number of output neurons
  * @param batch_size - Number of samples in batch
  */
 __global__ void updateWeights(float* weights, const float* input, const float* error, 
                             float learning_rate, int input_size, 
                             int output_size, int batch_size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     int input_idx = idx / output_size;
     int output_idx = idx % output_size;
     
     if (input_idx < input_size && output_idx < output_size) {
         float gradient_sum = 0.0f;
         for (int b = 0; b < batch_size; b++) {
             gradient_sum += input[b * input_size + input_idx] * 
                             error[b * output_size + output_idx];
         }
         weights[input_idx * output_size + output_idx] -= learning_rate * gradient_sum / batch_size;
     }
 }
 
 /**
  * CUDA kernel for updating bias values based on gradients
  * 
  * @param biases - Bias vector to update
  * @param error - Error gradient
  * @param learning_rate - Learning rate for gradient descent
  * @param size - Number of neurons (size of bias vector)
  * @param batch_size - Number of samples in batch
  */
 __global__ void updateBiases(float* biases, const float* error, 
                            float learning_rate, int size, int batch_size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (idx < size) {
         float gradient_sum = 0.0f;
         for (int b = 0; b < batch_size; b++) {
             gradient_sum += error[b * size + idx];
         }
         biases[idx] -= learning_rate * gradient_sum / batch_size;
     }
 }
 
 /**
  * CUDA kernel for adding biases to layer outputs
  * 
  * @param output - Layer output to which biases will be added
  * @param biases - Bias vector
  * @param batch_size - Number of samples in batch
  * @param output_size - Number of neurons in layer
  */
 __global__ void addBiases(float* output, const float* biases, 
                          int batch_size, int output_size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     int batch_idx = idx / output_size;
     int output_idx = idx % output_size;
     
     if (batch_idx < batch_size && output_idx < output_size) {
         output[idx] += biases[output_idx];
     }
 }
 
 /**
  * Read an integer from big-endian format in binary file
  * 
  * @param file - File pointer
  * @return The integer read
  */
 int readInt(FILE* file) {
     unsigned char buffer[4];
     if (fread(buffer, 1, 4, file) != 4) {
         fprintf(stderr, "Error reading file\n");
         exit(EXIT_FAILURE);
     }
     
     // Convert from big-endian format (MNIST format)
     return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
 }
 
 /**
  * Function to read MNIST data from binary files
  * 
  * This function properly parses the IDX file format used by MNIST
  * 
  * @param imageFile - Path to MNIST image file (.idx3-ubyte)
  * @param labelFile - Path to MNIST label file (.idx1-ubyte)
  * @param images - Output array for image data
  * @param labels - Output array for label data
  * @param numSamples - Number of samples to read
  */
 void loadMNISTData(const char* imageFile, const char* labelFile, 
                   float* images, unsigned char* labels, int numSamples) {
     // Open image file
     FILE* imgFile = fopen(imageFile, "rb");
     if (!imgFile) {
         fprintf(stderr, "Error opening image file: %s\n", imageFile);
         exit(EXIT_FAILURE);
     }
     
     // Read image file header
     int magicNumber = readInt(imgFile);
     int numImages = readInt(imgFile);
     int numRows = readInt(imgFile);
     int numCols = readInt(imgFile);
     
     // Check magic number for image file (should be 2051)
     if (magicNumber != 2051) {
         fprintf(stderr, "Invalid magic number in image file: %d\n", magicNumber);
         fclose(imgFile);
         exit(EXIT_FAILURE);
     }
     
     // Check dimensions
     if (numRows * numCols != INPUT_SIZE) {
         fprintf(stderr, "Unexpected image dimensions: %dx%d\n", numRows, numCols);
         fclose(imgFile);
         exit(EXIT_FAILURE);
     }
     
     // Check if we have enough images
     if (numImages < numSamples) {
         fprintf(stderr, "Not enough images in file. Requested: %d, Available: %d\n", 
                 numSamples, numImages);
         fclose(imgFile);
         exit(EXIT_FAILURE);
     }
     
     // Open label file
     FILE* lblFile = fopen(labelFile, "rb");
     if (!lblFile) {
         fprintf(stderr, "Error opening label file: %s\n", labelFile);
         fclose(imgFile);
         exit(EXIT_FAILURE);
     }
     
     // Read label file header
     magicNumber = readInt(lblFile);
     int numLabels = readInt(lblFile);
     
     // Check magic number for label file (should be 2049)
     if (magicNumber != 2049) {
         fprintf(stderr, "Invalid magic number in label file: %d\n", magicNumber);
         fclose(imgFile);
         fclose(lblFile);
         exit(EXIT_FAILURE);
     }
     
     // Check if we have enough labels
     if (numLabels < numSamples) {
         fprintf(stderr, "Not enough labels in file. Requested: %d, Available: %d\n", 
                 numSamples, numLabels);
         fclose(imgFile);
         fclose(lblFile);
         exit(EXIT_FAILURE);
     }
     
     // Buffer for reading image pixels
     unsigned char* pixelBuffer = (unsigned char*)malloc(INPUT_SIZE);
     if (!pixelBuffer) {
         fprintf(stderr, "Failed to allocate memory for pixel buffer\n");
         fclose(imgFile);
         fclose(lblFile);
         exit(EXIT_FAILURE);
     }
     
     // Read images and labels
     printf("Loading %d MNIST samples...\n", numSamples);
     for (int i = 0; i < numSamples; i++) {
         // Read pixels for one image
         if (fread(pixelBuffer, 1, INPUT_SIZE, imgFile) != INPUT_SIZE) {
             fprintf(stderr, "Error reading image %d\n", i);
             free(pixelBuffer);
             fclose(imgFile);
             fclose(lblFile);
             exit(EXIT_FAILURE);
         }
         
         // Normalize pixel values to [0, 1] range and store in the images array
         for (int j = 0; j < INPUT_SIZE; j++) {
             images[i * INPUT_SIZE + j] = pixelBuffer[j] / 255.0f;
         }
         
         // Read one label
         if (fread(&labels[i], 1, 1, lblFile) != 1) {
             fprintf(stderr, "Error reading label %d\n", i);
             free(pixelBuffer);
             fclose(imgFile);
             fclose(lblFile);
             exit(EXIT_FAILURE);
         }
     }
     
     // Clean up
     free(pixelBuffer);
     fclose(imgFile);
     fclose(lblFile);
     
     printf("Successfully loaded %d MNIST samples\n", numSamples);
 }
 
 /**
  * Initialize weights using Xavier/Glorot initialization
  * This helps prevent vanishing/exploding gradients
  * 
  * @param weights - Weight matrix to initialize
  * @param inputSize - Number of input neurons
  * @param outputSize - Number of output neurons
  */
 void initializeWeights(float* weights, int inputSize, int outputSize) {
     // Xavier initialization scale
     float xavier_limit = sqrtf(6.0f / (inputSize + outputSize));
     
     for (int i = 0; i < inputSize * outputSize; i++) {
         // Generate random values between -xavier_limit and xavier_limit
         weights[i] = ((float)rand() / RAND_MAX) * 2.0f * xavier_limit - xavier_limit;
     }
 }
 
 /**
  * Initialize biases to zeros
  * 
  * @param biases - Bias vector to initialize
  * @param size - Number of neurons (size of bias vector)
  */
 void initializeBiases(float* biases, int size) {
     memset(biases, 0, size * sizeof(float));
 }
 
 /**
  * Print summary information about a network layer
  * 
  * @param weights - Weight matrix
  * @param inputSize - Number of input neurons
  * @param outputSize - Number of output neurons
  * @param layerName - Name of the layer for display
  */
 void printLayerInfo(const float* weights, int inputSize, int outputSize, const char* layerName) {
     float sum = 0.0f, min = weights[0], max = weights[0];
     
     for (int i = 0; i < inputSize * outputSize; i++) {
         sum += weights[i];
         min = fminf(min, weights[i]);
         max = fmaxf(max, weights[i]);
     }
     
     float mean = sum / (inputSize * outputSize);
     
     printf("%s (%d -> %d): min=%.4f, max=%.4f, mean=%.4f\n", 
            layerName, inputSize, outputSize, min, max, mean);
 }
 
 /**
  * Train the neural network on MNIST data
  * 
  * @param trainImages - Training images (flattened)
  * @param trainLabels - Training labels
  * @param weights1 - Weights for input->hidden layer
  * @param weights2 - Weights for hidden->output layer
  * @param biases1 - Biases for hidden layer
  * @param biases2 - Biases for output layer
  */
 void trainNetwork(float* trainImages, unsigned char* trainLabels, 
                  float* weights1, float* weights2, float* biases1, float* biases2) {
     // Allocate device memory
     float *d_images, *d_weights1, *d_weights2, *d_biases1, *d_biases2;
     unsigned char *d_labels;
     float *d_hidden_preact, *d_hidden_output, *d_output_preact, *d_output;
     float *d_output_error, *d_hidden_error, *d_hidden_deriv, *d_loss;
     
     // Allocate memory for network parameters
     CUDA_CHECK(cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_biases1, HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_biases2, OUTPUT_SIZE * sizeof(float)));
     
     // Allocate memory for batch data
     CUDA_CHECK(cudaMalloc(&d_images, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(unsigned char)));
     
     // Allocate memory for intermediate values
     CUDA_CHECK(cudaMalloc(&d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_hidden_output, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_output_preact, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     
     // Allocate memory for backpropagation
     CUDA_CHECK(cudaMalloc(&d_output_error, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_hidden_error, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_hidden_deriv, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
     
     // Copy network parameters to device
     CUDA_CHECK(cudaMemcpy(d_weights1, weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_weights2, weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_biases1, biases1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_biases2, biases2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     
     // Define grid and block dimensions for different kernels
     // For 2D matrix multiplication
     dim3 blockSize2D(16, 16);
     dim3 gridSize2D_hidden((HIDDEN_SIZE + blockSize2D.x - 1) / blockSize2D.x, 
                          (BATCH_SIZE + blockSize2D.y - 1) / blockSize2D.y);
     dim3 gridSize2D_output((OUTPUT_SIZE + blockSize2D.x - 1) / blockSize2D.x, 
                           (BATCH_SIZE + blockSize2D.y - 1) / blockSize2D.y);
     
     // For 1D operations
     int blockSize1D = 256;
     int gridSize1D_hidden = (BATCH_SIZE * HIDDEN_SIZE + blockSize1D - 1) / blockSize1D;
     int gridSize1D_output = (BATCH_SIZE * OUTPUT_SIZE + blockSize1D - 1) / blockSize1D;
     int gridSize1D_batch = (BATCH_SIZE + blockSize1D - 1) / blockSize1D;
     int gridSize1D_w1 = (INPUT_SIZE * HIDDEN_SIZE + blockSize1D - 1) / blockSize1D;
     int gridSize1D_w2 = (HIDDEN_SIZE * OUTPUT_SIZE + blockSize1D - 1) / blockSize1D;
     
     // Training loop
     for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
         float total_loss = 0.0f;
         
         // Process mini-batches
         int num_batches = MNIST_TRAIN_SIZE / BATCH_SIZE;
         for (int batch = 0; batch < num_batches; batch++) {
             // Copy batch data to device
             CUDA_CHECK(cudaMemcpy(d_images, 
                                  &trainImages[batch * BATCH_SIZE * INPUT_SIZE], 
                                  BATCH_SIZE * INPUT_SIZE * sizeof(float), 
                                  cudaMemcpyHostToDevice));
             CUDA_CHECK(cudaMemcpy(d_labels, 
                                  &trainLabels[batch * BATCH_SIZE], 
                                  BATCH_SIZE * sizeof(unsigned char), 
                                  cudaMemcpyHostToDevice));
             
             // Initialize loss to 0
             float zero = 0.0f;
             CUDA_CHECK(cudaMemcpy(d_loss, &zero, sizeof(float), cudaMemcpyHostToDevice));
             
             // Forward pass
             
             // 1. Hidden layer: input -> hidden
             matrixMultiply<<<gridSize2D_hidden, blockSize2D>>>(
                 d_images, d_weights1, d_hidden_preact, 
                 BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
             
             // Add biases to hidden layer preactivation
             addBiases<<<gridSize1D_hidden, blockSize1D>>>(
                 d_hidden_preact, d_biases1, BATCH_SIZE, HIDDEN_SIZE);
             
             // Apply ReLU activation
             reluActivation<<<gridSize1D_hidden, blockSize1D>>>(
                 d_hidden_preact, d_hidden_output, BATCH_SIZE * HIDDEN_SIZE);
             
             // 2. Output layer: hidden -> output
             matrixMultiply<<<gridSize2D_output, blockSize2D>>>(
                 d_hidden_output, d_weights2, d_output_preact, 
                 BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
             
             // Add biases to output layer preactivation
             addBiases<<<gridSize1D_output, blockSize1D>>>(
                 d_output_preact, d_biases2, BATCH_SIZE, OUTPUT_SIZE);
             
             // Apply softmax activation
             softmaxActivation<<<gridSize1D_batch, blockSize1D>>>(
                 d_output_preact, d_output, BATCH_SIZE, OUTPUT_SIZE);
             
             // Calculate loss
             crossEntropyLoss<<<gridSize1D_batch, blockSize1D>>>(
                 d_output, d_labels, d_loss, BATCH_SIZE, OUTPUT_SIZE);
             
             // Backward pass
             
             // 1. Output layer error
             outputLayerError<<<gridSize1D_output, blockSize1D>>>(
                 d_output, d_labels, d_output_error, BATCH_SIZE, OUTPUT_SIZE);
             
             // 2. Calculate ReLU derivative for hidden layer
             reluDerivative<<<gridSize1D_hidden, blockSize1D>>>(
                 d_hidden_preact, d_hidden_deriv, BATCH_SIZE * HIDDEN_SIZE);
             
             // 3. Hidden layer error
             hiddenLayerError<<<gridSize1D_hidden, blockSize1D>>>(
                 d_output_error, d_weights2, d_hidden_deriv, d_hidden_error,
                 BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
             
             // Update weights and biases
             
             // 1. Update output layer weights
             updateWeights<<<gridSize1D_w2, blockSize1D>>>(
                 d_weights2, d_hidden_output, d_output_error,
                 LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);
             
             // 2. Update hidden layer weights
             updateWeights<<<gridSize1D_w1, blockSize1D>>>(
                 d_weights1, d_images, d_hidden_error,
                 LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE);
             
             // 3. Update output layer biases
             updateBiases<<<(OUTPUT_SIZE + blockSize1D - 1) / blockSize1D, blockSize1D>>>(
                 d_biases2, d_output_error, LEARNING_RATE, OUTPUT_SIZE, BATCH_SIZE);
             
             // 4. Update hidden layer biases
             updateBiases<<<(HIDDEN_SIZE + blockSize1D - 1) / blockSize1D, blockSize1D>>>(
                 d_biases1, d_hidden_error, LEARNING_RATE, HIDDEN_SIZE, BATCH_SIZE);
             
             // Copy loss back to host
             float batch_loss;
             CUDA_CHECK(cudaMemcpy(&batch_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
             batch_loss /= BATCH_SIZE;  // Average loss per sample
             total_loss += batch_loss;
             
             // Check for CUDA errors
             cudaError_t error = cudaGetLastError();
             if (error != cudaSuccess) {
                 fprintf(stderr, "CUDA error in batch %d: %s\n", batch, cudaGetErrorString(error));
                 exit(EXIT_FAILURE);
             }
         }
         
         // Print epoch results
         printf("Epoch %d/%d: Average Loss = %.4f\n", 
                epoch + 1, NUM_EPOCHS, total_loss / num_batches);
     }
     
     // Copy updated parameters back to host
     CUDA_CHECK(cudaMemcpy(weights1, d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
     CUDA_CHECK(cudaMemcpy(weights2, d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
     CUDA_CHECK(cudaMemcpy(biases1, d_biases1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
     CUDA_CHECK(cudaMemcpy(biases2, d_biases2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
     
     // Free device memory
     CUDA_CHECK(cudaFree(d_weights1));
     CUDA_CHECK(cudaFree(d_weights2));
     CUDA_CHECK(cudaFree(d_biases1));
     CUDA_CHECK(cudaFree(d_biases2));
     CUDA_CHECK(cudaFree(d_images));
     CUDA_CHECK(cudaFree(d_labels));
     CUDA_CHECK(cudaFree(d_hidden_preact));
     CUDA_CHECK(cudaFree(d_hidden_output));
     CUDA_CHECK(cudaFree(d_output_preact));
     CUDA_CHECK(cudaFree(d_output));
     CUDA_CHECK(cudaFree(d_output_error));
     CUDA_CHECK(cudaFree(d_hidden_error));
     CUDA_CHECK(cudaFree(d_hidden_deriv));
     CUDA_CHECK(cudaFree(d_loss));
 }
 
 /**
  * Test the neural network on MNIST data
  * 
  * @param testImages - Test images (flattened)
  * @param testLabels - Test labels
  * @param weights1 - Weights for input->hidden layer
  * @param weights2 - Weights for hidden->output layer
  * @param biases1 - Biases for hidden layer
  * @param biases2 - Biases for output layer
  */
 void testNetwork(float* testImages, unsigned char* testLabels, 
                 float* weights1, float* weights2, float* biases1, float* biases2) {
     // Allocate device memory
     float *d_images, *d_weights1, *d_weights2, *d_biases1, *d_biases2;
     unsigned char *d_labels;
     float *d_hidden_preact, *d_hidden_output, *d_output_preact, *d_output;
     
     // Allocate memory for network parameters
     CUDA_CHECK(cudaMalloc(&d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_biases1, HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_biases2, OUTPUT_SIZE * sizeof(float)));
     
     // Allocate memory for batch data
     CUDA_CHECK(cudaMalloc(&d_images, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(unsigned char)));
     
     // Allocate memory for intermediate values
     CUDA_CHECK(cudaMalloc(&d_hidden_preact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_hidden_output, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_output_preact, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     
     // Copy network parameters to device
     CUDA_CHECK(cudaMemcpy(d_weights1, weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_weights2, weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_biases1, biases1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_biases2, biases2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     
     // Define grid and block dimensions for different kernels
     dim3 blockSize2D(16, 16);
     dim3 gridSize2D_hidden((HIDDEN_SIZE + blockSize2D.x - 1) / blockSize2D.x, 
                          (BATCH_SIZE + blockSize2D.y - 1) / blockSize2D.y);
     dim3 gridSize2D_output((OUTPUT_SIZE + blockSize2D.x - 1) / blockSize2D.x, 
                           (BATCH_SIZE + blockSize2D.y - 1) / blockSize2D.y);
     
     int blockSize1D = 256;
     int gridSize1D_hidden = (BATCH_SIZE * HIDDEN_SIZE + blockSize1D - 1) / blockSize1D;
     int gridSize1D_output = (BATCH_SIZE * OUTPUT_SIZE + blockSize1D - 1) / blockSize1D;
     int gridSize1D_batch = (BATCH_SIZE + blockSize1D - 1) / blockSize1D;
     
     // Test variables
     int total_correct = 0;
     float* h_output = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
     
     if (h_output == NULL) {
         fprintf(stderr, "Error: Memory allocation failed for h_output\n");
         exit(EXIT_FAILURE);
     }
     
     // Create confusion matrix
     int confusion_matrix[OUTPUT_SIZE][OUTPUT_SIZE] = {0};
     
     // Process mini-batches
     int num_batches = MNIST_TEST_SIZE / BATCH_SIZE;
     for (int batch = 0; batch < num_batches; batch++) {
         // Copy batch data to device
         CUDA_CHECK(cudaMemcpy(d_images, 
                              &testImages[batch * BATCH_SIZE * INPUT_SIZE], 
                              BATCH_SIZE * INPUT_SIZE * sizeof(float), 
                              cudaMemcpyHostToDevice));
         CUDA_CHECK(cudaMemcpy(d_labels, 
                              &testLabels[batch * BATCH_SIZE], 
                              BATCH_SIZE * sizeof(unsigned char), 
                              cudaMemcpyHostToDevice));
         
         // Forward pass
         
         // 1. Hidden layer: input -> hidden
         matrixMultiply<<<gridSize2D_hidden, blockSize2D>>>(
             d_images, d_weights1, d_hidden_preact, 
             BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
         
         // Add biases to hidden layer preactivation
         addBiases<<<gridSize1D_hidden, blockSize1D>>>(
             d_hidden_preact, d_biases1, BATCH_SIZE, HIDDEN_SIZE);
         
         // Apply ReLU activation
         reluActivation<<<gridSize1D_hidden, blockSize1D>>>(
             d_hidden_preact, d_hidden_output, BATCH_SIZE * HIDDEN_SIZE);
         
         // 2. Output layer: hidden -> output
         matrixMultiply<<<gridSize2D_output, blockSize2D>>>(
             d_hidden_output, d_weights2, d_output_preact, 
             BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
         
         // Add biases to output layer preactivation
         addBiases<<<gridSize1D_output, blockSize1D>>>(
             d_output_preact, d_biases2, BATCH_SIZE, OUTPUT_SIZE);
         
         // Apply softmax activation
         softmaxActivation<<<gridSize1D_batch, blockSize1D>>>(
             d_output_preact, d_output, BATCH_SIZE, OUTPUT_SIZE);
         
         // Copy output back to host
         CUDA_CHECK(cudaMemcpy(h_output, d_output, 
                              BATCH_SIZE * OUTPUT_SIZE * sizeof(float), 
                              cudaMemcpyDeviceToHost));
         
         // Count correct predictions and update confusion matrix
         for (int i = 0; i < BATCH_SIZE; i++) {
             // Find predicted class (maximum probability)
             int predicted_class = 0;
             float max_prob = h_output[i * OUTPUT_SIZE];
             
             for (int j = 1; j < OUTPUT_SIZE; j++) {
                 if (h_output[i * OUTPUT_SIZE + j] > max_prob) {
                     max_prob = h_output[i * OUTPUT_SIZE + j];
                     predicted_class = j;
                 }
             }
             
             // Get true label
             int true_label = testLabels[batch * BATCH_SIZE + i];
             
             // Update confusion matrix
             confusion_matrix[true_label][predicted_class]++;
             
             // Check if prediction is correct
             if (predicted_class == true_label) {
                 total_correct++;
             }
         }
     }
     
     // Print test accuracy
     float accuracy = (float)total_correct / MNIST_TEST_SIZE * 100.0f;
     printf("Test Accuracy: %.2f%% (%d/%d)\n", 
            accuracy, total_correct, MNIST_TEST_SIZE);
     
     // Print confusion matrix (optional)
     printf("\nConfusion Matrix:\n");
     printf("    ");
     for (int i = 0; i < OUTPUT_SIZE; i++) {
         printf("%4d ", i);
     }
     printf("\n");
     
     for (int i = 0; i < OUTPUT_SIZE; i++) {
         printf("%2d: ", i);
         for (int j = 0; j < OUTPUT_SIZE; j++) {
             printf("%4d ", confusion_matrix[i][j]);
         }
         printf("\n");
     }
     
     // Print per-class accuracy
     printf("\nPer-class Accuracy:\n");
     for (int i = 0; i < OUTPUT_SIZE; i++) {
         int class_total = 0;
         for (int j = 0; j < OUTPUT_SIZE; j++) {
             class_total += confusion_matrix[i][j];
         }
         float class_accuracy = (float)confusion_matrix[i][i] / class_total * 100.0f;
         printf("Class %d: %.2f%%\n", i, class_accuracy);
     }
     
     // Free memory
     free(h_output);
     CUDA_CHECK(cudaFree(d_weights1));
     CUDA_CHECK(cudaFree(d_weights2));
     CUDA_CHECK(cudaFree(d_biases1));
     CUDA_CHECK(cudaFree(d_biases2));
     CUDA_CHECK(cudaFree(d_images));
     CUDA_CHECK(cudaFree(d_labels));
     CUDA_CHECK(cudaFree(d_hidden_preact));
     CUDA_CHECK(cudaFree(d_hidden_output));
     CUDA_CHECK(cudaFree(d_output_preact));
     CUDA_CHECK(cudaFree(d_output));
 }
 
 /**
  * Main function to run the MLP for MNIST classification
  */
 int main() {
     // Set random seed for reproducibility
     srand(time(NULL));
     
     // Define file paths for MNIST dataset
     const char* trainImagesFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-images.idx3-ubyte";
     const char* trainLabelsFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-labels.idx1-ubyte";
     const char* testImagesFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-images.idx3-ubyte";
     const char* testLabelsFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-labels.idx1-ubyte";
     
     // Allocate memory for MNIST data
     float* trainImages = (float*)malloc(MNIST_TRAIN_SIZE * INPUT_SIZE * sizeof(float));
     unsigned char* trainLabels = (unsigned char*)malloc(MNIST_TRAIN_SIZE * sizeof(unsigned char));
     float* testImages = (float*)malloc(MNIST_TEST_SIZE * INPUT_SIZE * sizeof(float));
     unsigned char* testLabels = (unsigned char*)malloc(MNIST_TEST_SIZE * sizeof(unsigned char));
     
     // Check memory allocation
     if (!trainImages || !trainLabels || !testImages || !testLabels) {
         fprintf(stderr, "Error: Memory allocation failed for dataset\n");
         // Free any successfully allocated memory
         if (trainImages) free(trainImages);
         if (trainLabels) free(trainLabels);
         if (testImages) free(testImages);
         if (testLabels) free(testLabels);
         return EXIT_FAILURE;
     }
     
     // Load MNIST data
     printf("Loading MNIST data...\n");
     loadMNISTData(trainImagesFile, trainLabelsFile, 
                  trainImages, trainLabels, MNIST_TRAIN_SIZE);
     loadMNISTData(testImagesFile, testLabelsFile, 
                  testImages, testLabels, MNIST_TEST_SIZE);
     
     // Allocate memory for network parameters
     float* weights1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
     float* weights2 = (float*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
     float* biases1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
     float* biases2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
     
     // Check memory allocation
     if (!weights1 || !weights2 || !biases1 || !biases2) {
         fprintf(stderr, "Error: Memory allocation failed for network parameters\n");
         // Free any successfully allocated memory
         if (trainImages) free(trainImages);
         if (trainLabels) free(trainLabels);
         if (testImages) free(testImages);
         if (testLabels) free(testLabels);
         if (weights1) free(weights1);
         if (weights2) free(weights2);
         if (biases1) free(biases1);
         if (biases2) free(biases2);
         return EXIT_FAILURE;
     }
     
     // Initialize network parameters
     printf("Initializing network parameters...\n");
     initializeWeights(weights1, INPUT_SIZE, HIDDEN_SIZE);
     initializeWeights(weights2, HIDDEN_SIZE, OUTPUT_SIZE);
     initializeBiases(biases1, HIDDEN_SIZE);
     initializeBiases(biases2, OUTPUT_SIZE);
     
     // Print initial network summary
     printLayerInfo(weights1, INPUT_SIZE, HIDDEN_SIZE, "Hidden Layer");
     printLayerInfo(weights2, HIDDEN_SIZE, OUTPUT_SIZE, "Output Layer");
     
     printf("Training MLP for MNIST digit classification...\n");
     
     // Create CUDA events for timing
     cudaEvent_t start, stop;
     CUDA_CHECK(cudaEventCreate(&start));
     CUDA_CHECK(cudaEventCreate(&stop));
     
     // Record start time
     CUDA_CHECK(cudaEventRecord(start, 0));
     
     // Train network
     trainNetwork(trainImages, trainLabels, weights1, weights2, biases1, biases2);
     
     // Record stop time
     CUDA_CHECK(cudaEventRecord(stop, 0));
     CUDA_CHECK(cudaEventSynchronize(stop));
     
     // Calculate training time
     float training_time;
     CUDA_CHECK(cudaEventElapsedTime(&training_time, start, stop));
     printf("Training completed in %.2f seconds\n", training_time / 1000.0f);
     
     // Print trained network summary
     printLayerInfo(weights1, INPUT_SIZE, HIDDEN_SIZE, "Trained Hidden Layer");
     printLayerInfo(weights2, HIDDEN_SIZE, OUTPUT_SIZE, "Trained Output Layer");
     
     printf("Testing network...\n");
     
     // Test network
     testNetwork(testImages, testLabels, weights1, weights2, biases1, biases2);
     
     // Free memory
     free(trainImages);
     free(trainLabels);
     free(testImages);
     free(testLabels);
     free(weights1);
     free(weights2);
     free(biases1);
     free(biases2);
     CUDA_CHECK(cudaEventDestroy(start));
     CUDA_CHECK(cudaEventDestroy(stop));
     
     // Print memory usage and CUDA device info
     size_t free_memory, total_memory;
     CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
     printf("\nCUDA Memory: %.2f MB free / %.2f MB total\n", 
            free_memory / (1024.0f * 1024.0f), 
            total_memory / (1024.0f * 1024.0f));
     
     cudaDeviceProp deviceProp;
     CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
     printf("GPU: %s\n", deviceProp.name);
     
     printf("Done!\n");
     
     return EXIT_SUCCESS;
 }
 
 /**
  * Alternative implementation considerations:
  * 
  * 1. CUDA Optimizations:
  *    - Use shared memory for matrix multiplication to improve performance
  *    - Implement cuBLAS for matrix operations instead of custom kernels
  *    - Use cuDNN for neural network operations
  * 
  * 2. Architecture Variations:
  *    - Add more hidden layers for better accuracy
  *    - Use different activation functions (e.g., Leaky ReLU, ELU)
  *    - Implement batch normalization for faster convergence
  *    - Add dropout for regularization
  * 
  * 3. Performance:
  *    - Use half-precision (FP16) for faster computation
  *    - Implement more sophisticated optimization algorithms (Adam, RMSProp)
  *    - Add learning rate scheduling for better convergence
  * 
  * 4. Functionality:
  *    - Add early stopping based on validation accuracy
  *    - Implement data augmentation to improve generalization
  *    - Add checkpoint saving/loading for long training runs
  */