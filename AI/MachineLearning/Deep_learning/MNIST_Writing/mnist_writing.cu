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
 #define CUDA_CHECK(Call) \
 do { \
     cudaError_t Error = Call; \
     if (Error != cudaSuccess) { \
         fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                 __FILE__, __LINE__, cudaGetErrorString(Error)); \
         exit(EXIT_FAILURE); \
     } \
 } while(0)
 
 // Function declarations
 void LoadMNISTData(const char* ImageFile, const char* LabelFile, 
                   float* Images, unsigned char* Labels, int NumSamples);
 void InitializeWeights(float* Weights, int InputSize, int OutputSize);
 void InitializeBiases(float* Biases, int size);
 void TrainNetwork(float* TrainImages, unsigned char* TrainLabels, 
                  float* Weights1, float* Weights2, float* Biases1, float* Biases2);
 void TestNetwork(float* TestImages, unsigned char* TestLabels, 
                 float* Weights1, float* Weights2, float* Biases1, float* Biases2);
 void PrintLayerInfo(const float* Weights, int InputSize, int OutputSize, const char* LayerName);
 
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
 __global__ void MatrixMultiply(const float* A, const float* B, float* C, 
                               int A_rows, int A_cols, int B_cols) {
     // Calculate global thread indices
     int Row = blockIdx.y * blockDim.y + threadIdx.y;
     int Col = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Check if thread is within bounds
     if (Row < A_rows && Col < B_cols) {
         float Sum = 0.0f;
         for (int i = 0; i < A_cols; i++) {
             Sum += A[Row * A_cols + i] * B[i * B_cols + Col];
         }
         C[Row * B_cols + Col] = Sum;
     }
 }
 
 /**
  * CUDA kernel for ReLU activation function: output = max(0, input)
  * 
  * @param input - Input array
  * @param output - Output array (result of ReLU operation)
  * @param size - Size of the input and output arrays
  */
 __global__ void relu_activation(const float* Input, float* Output, int size) {
     int Idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (Idx < size) {
         Output[Idx] = fmaxf(0.0f, Input[Idx]);
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
 __global__ void ReluDerivative(const float* Input, float* Output, int size) {
     int Idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (Idx < size) {
         Output[Idx] = (Input[Idx] > 0.0f) ? 1.0f : 0.0f;
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
 __global__ void SoftmaxActivation(const float* Input, float* Output, 
                                  int BatchSize, int OutputSize) {
     int BatchIdx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (BatchIdx < BatchSize) {
         // Find max for numerical stability
         float MaxVal = Input[BatchIdx * OutputSize];
         for (int i = 1; i < OutputSize; i++) {
             MaxVal = fmaxf(MaxVal, Input[BatchIdx * OutputSize + i]);
         }
         
         // Calculate exp(x - max) and sum
         float Sum = 0.0f;
         for (int i = 0; i < OutputSize; i++) {
             Output[BatchIdx * OutputSize + i] = expf(Input[BatchIdx * OutputSize + i] - MaxVal);
             Sum += Output[BatchIdx * OutputSize + i];
         }
         
         // Normalize by sum
         for (int i = 0; i < OutputSize; i++) {
             Output[BatchIdx * OutputSize + i] /= Sum;
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
 __global__ void CrossEntropyLoss(const float* Predictions, const unsigned char* Labels, 
                                 float* Loss, int BatchSize, int NumClasses) {
     int BatchIdx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (BatchIdx < BatchSize) {
         int Label = Labels[BatchIdx];
         // Ensure valid label (defensive programming)
         if (Label >= 0 && Label < NumClasses) {
             // Clip prediction to avoid log(0)
             float PredictedProb = fmaxf(Predictions[BatchIdx * NumClasses + Label], 1e-15f);
             atomicAdd(Loss, -logf(PredictedProb));
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
 __global__ void OutputLayerError(const float* Predictions, const unsigned char* Labels, 
                                float* Error, int BatchSize, int NumClasses) {
     int Idx = blockIdx.x * blockDim.x + threadIdx.x;
     int BatchIdx = Idx / NumClasses;
     int ClassIdx = Idx % NumClasses;
     
     if (BatchIdx < BatchSize && ClassIdx < NumClasses) {
         // Calculate softmax gradient (one-hot - predicted)
         if (Labels[BatchIdx] == ClassIdx) {
             Error[Idx] = Predictions[Idx] - 1.0f;
         } else {
             Error[Idx] = Predictions[Idx];
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
 __global__ void HiddenLayerError(const float* OutputError, const float* Weights, 
                                const float* ReluDerivative, float* HiddenError,
                                int BatchSize, int HiddenSize, int OutputSize) {
     int Idx = blockIdx.x * blockDim.x + threadIdx.x;
     int BatchIdx = Idx / HiddenSize;
     int HiddenIdx = Idx % HiddenSize;
     
     if (BatchIdx < BatchSize && HiddenIdx < HiddenSize) {
         float ErrorSum = 0.0f;
         for (int i = 0; i < OutputSize; i++) {
             ErrorSum += OutputError[BatchIdx * OutputSize + i] * 
                          Weights[HiddenIdx * OutputSize + i];
         }
         HiddenError[Idx] = ErrorSum * ReluDerivative[Idx];
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
 __global__ void UpdateWeights(float* Weights, const float* Input, const float* Error, 
                             float LearningRate, int InputSize, 
                             int OutputSize, int BatchSize) {
     int Idx = blockIdx.x * blockDim.x + threadIdx.x;
     int InputIdx = Idx / OutputSize;
     int OutputIdx = Idx % OutputSize;
     
     if (InputIdx < InputSize && OutputIdx < OutputSize) {
         float GradientSum = 0.0f;
         for (int b = 0; b < BatchSize; b++) {
             GradientSum += Input[b * InputSize + InputIdx] * 
                             Error[b * OutputSize + OutputIdx];
         }
         Weights[InputIdx * OutputSize + OutputIdx] -= LearningRate * GradientSum / BatchSize;
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
 __global__ void UpdateBiases(float* Biases, const float* Error, 
                            float LearningRate, int size, int BatchSize) {
     int Idx = blockIdx.x * blockDim.x + threadIdx.x;
     
     if (Idx < size) {
         float GradientSum = 0.0f;
         for (int b = 0; b < BatchSize; b++) {
             GradientSum += Error[b * size + Idx];
         }
         Biases[Idx] -= LearningRate * GradientSum / BatchSize;
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
 __global__ void AddBiases(float* Output, const float* Biases, 
                          int BatchSize, int OutputSize) {
     int Idx = blockIdx.x * blockDim.x + threadIdx.x;
     int BatchIdx = Idx / OutputSize;
     int OutputIdx = Idx % OutputSize;
     
     if (BatchIdx < BatchSize && OutputIdx < OutputSize) {
         Output[Idx] += Biases[OutputIdx];
     }
 }
 
 /**
  * Read an integer from big-endian format in binary file
  * 
  * @param file - File pointer
  * @return The integer read
  */
 int ReadInt(FILE* File) {
     unsigned char Buffer[4];
     if (fread(Buffer, 1, 4, File) != 4) {
         fprintf(stderr, "Error reading file\n");
         exit(EXIT_FAILURE);
     }
     
     // Convert from big-endian format (MNIST format)
     return (Buffer[0] << 24) | (Buffer[1] << 16) | (Buffer[2] << 8) | Buffer[3];
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
 void LoadMNISTData(const char* ImageFile, const char* LabelFile, 
                   float* Images, unsigned char* Labels, int NumSamples) {
     // Open image file
     FILE* ImgFile = fopen(ImageFile, "rb");
     if (!ImgFile) {
         fprintf(stderr, "Error opening image file: %s\n", ImageFile);
         exit(EXIT_FAILURE);
     }
     
     // Read image file header
     int MagicNumber = ReadInt(ImgFile);
     int NumImages = ReadInt(ImgFile);
     int NumRows = ReadInt(ImgFile);
     int NumCols = ReadInt(ImgFile);
     
     // Check magic number for image file (should be 2051)
     if (MagicNumber != 2051) {
         fprintf(stderr, "Invalid magic number in image file: %d\n", MagicNumber);
         fclose(ImgFile);
         exit(EXIT_FAILURE);
     }
     
     // Check dimensions
     if (NumRows * NumCols != INPUT_SIZE) {
         fprintf(stderr, "Unexpected image dimensions: %dx%d\n", NumRows, NumCols);
         fclose(ImgFile);
         exit(EXIT_FAILURE);
     }
     
     // Check if we have enough images
     if (NumImages < NumSamples) {
         fprintf(stderr, "Not enough images in file. Requested: %d, Available: %d\n", 
                 NumSamples, NumImages);
         fclose(ImgFile);
         exit(EXIT_FAILURE);
     }
     
     // Open label file
     FILE* LblFile = fopen(LabelFile, "rb");
     if (!LblFile) {
         fprintf(stderr, "Error opening label file: %s\n", LabelFile);
         fclose(ImgFile);
         exit(EXIT_FAILURE);
     }
     
     // Read label file header
     MagicNumber = ReadInt(LblFile);
     int NumLabels = ReadInt(LblFile);
     
     // Check magic number for label file (should be 2049)
     if (MagicNumber != 2049) {
         fprintf(stderr, "Invalid magic number in label file: %d\n", MagicNumber);
         fclose(ImgFile);
         fclose(LblFile);
         exit(EXIT_FAILURE);
     }
     
     // Check if we have enough labels
     if (NumLabels < NumSamples) {
         fprintf(stderr, "Not enough labels in file. Requested: %d, Available: %d\n", 
                 NumSamples, NumLabels);
         fclose(ImgFile);
         fclose(LblFile);
         exit(EXIT_FAILURE);
     }
     
     // Buffer for reading image pixels
     unsigned char* PixelBuffer = (unsigned char*)malloc(INPUT_SIZE);
     if (!PixelBuffer) {
         fprintf(stderr, "Failed to allocate memory for pixel buffer\n");
         fclose(ImgFile);
         fclose(LblFile);
         exit(EXIT_FAILURE);
     }
     
     // Read images and labels
     printf("Loading %d MNIST samples...\n", NumSamples);
     for (int i = 0; i < NumSamples; i++) {
         // Read pixels for one image
         if (fread(PixelBuffer, 1, INPUT_SIZE, ImgFile) != INPUT_SIZE) {
             fprintf(stderr, "Error reading image %d\n", i);
             free(PixelBuffer);
             fclose(ImgFile);
             fclose(LblFile);
             exit(EXIT_FAILURE);
         }
         
         // Normalize pixel values to [0, 1] range and store in the images array
         for (int j = 0; j < INPUT_SIZE; j++) {
             Images[i * INPUT_SIZE + j] = PixelBuffer[j] / 255.0f;
         }
         
         // Read one label
         if (fread(&Labels[i], 1, 1, LblFile) != 1) {
             fprintf(stderr, "Error reading label %d\n", i);
             free(PixelBuffer);
             fclose(ImgFile);
             fclose(LblFile);
             exit(EXIT_FAILURE);
         }
     }
     
     // Clean up
     free(PixelBuffer);
     fclose(ImgFile);
     fclose(LblFile);
     
     printf("Successfully loaded %d MNIST samples\n", NumSamples);
 }
 
 /**
  * Initialize weights using Xavier/Glorot initialization
  * This helps prevent vanishing/exploding gradients
  * 
  * @param weights - Weight matrix to initialize
  * @param inputSize - Number of input neurons
  * @param outputSize - Number of output neurons
  */
 void InitializeWeights(float* Weights, int InputSize, int OutputSize) {
     // Xavier initialization scale
     float XavierLimit = sqrtf(6.0f / (InputSize + OutputSize));
     
     for (int i = 0; i < InputSize * OutputSize; i++) {
         // Generate random values between -xavier_limit and xavier_limit
         Weights[i] = ((float)rand() / RAND_MAX) * 2.0f * XavierLimit - XavierLimit;
     }
 }
 
 /**
  * Initialize biases to zeros
  * 
  * @param biases - Bias vector to initialize
  * @param size - Number of neurons (size of bias vector)
  */
 void InitializeBiases(float* Biases, int size) {
     memset(Biases, 0, size * sizeof(float));
 }
 
 /**
  * Print summary information about a network layer
  * 
  * @param weights - Weight matrix
  * @param inputSize - Number of input neurons
  * @param outputSize - Number of output neurons
  * @param layerName - Name of the layer for display
  */
 void PrintLayerInfo(const float* Weights, int InputSize, int OutputSize, const char* LayerName) {
     float Sum = 0.0f, min = Weights[0], max = Weights[0];
     
     for (int i = 0; i < InputSize * OutputSize; i++) {
         Sum += Weights[i];
         min = fminf(min, Weights[i]);
         max = fmaxf(max, Weights[i]);
     }
     
     float Mean = Sum / (InputSize * OutputSize);
     
     printf("%s (%d -> %d): min=%.4f, max=%.4f, mean=%.4f\n", 
            LayerName, InputSize, OutputSize, min, max, Mean);
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
 void TrainNetwork(float* TrainImages, unsigned char* TrainLabels, 
                  float* Weights1, float* Weights2, float* Biases1, float* Biases2) {
     // Allocate device memory
     float *DImages, *DWeights1, *DWeights2, *DBiases1, *DBiases2;
     unsigned char *DLabels;
     float *DHiddenPreact, *DHiddenOutput, *DOutputPreact, *DOutput;
     float *DOutputError, *DHiddenError, *DHiddenDeriv, *DLoss;
     
     // Allocate memory for network parameters
     CUDA_CHECK(cudaMalloc(&DWeights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DWeights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DBiases1, HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DBiases2, OUTPUT_SIZE * sizeof(float)));
     
     // Allocate memory for batch data
     CUDA_CHECK(cudaMalloc(&DImages, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DLabels, BATCH_SIZE * sizeof(unsigned char)));
     
     // Allocate memory for intermediate values
     CUDA_CHECK(cudaMalloc(&DHiddenPreact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DHiddenOutput, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DOutputPreact, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DOutput, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     
     // Allocate memory for backpropagation
     CUDA_CHECK(cudaMalloc(&DOutputError, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DHiddenError, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DHiddenDeriv, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DLoss, sizeof(float)));
     
     // Copy network parameters to device
     CUDA_CHECK(cudaMemcpy(DWeights1, Weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(DWeights2, Weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(DBiases1, Biases1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(DBiases2, Biases2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     
     // Define grid and block dimensions for different kernels
     // For 2D matrix multiplication
     dim3 BlockSize2D(16, 16);
     dim3 gridSize2D_hidden((HIDDEN_SIZE + BlockSize2D.x - 1) / BlockSize2D.x, 
                          (BATCH_SIZE + BlockSize2D.y - 1) / BlockSize2D.y);
     dim3 gridSize2D_output((OUTPUT_SIZE + BlockSize2D.x - 1) / BlockSize2D.x, 
                           (BATCH_SIZE + BlockSize2D.y - 1) / BlockSize2D.y);
     
     // For 1D operations
     int BlockSize1D = 256;
     int gridSize1D_hidden = (BATCH_SIZE * HIDDEN_SIZE + BlockSize1D - 1) / BlockSize1D;
     int gridSize1D_output = (BATCH_SIZE * OUTPUT_SIZE + BlockSize1D - 1) / BlockSize1D;
     int gridSize1D_batch = (BATCH_SIZE + BlockSize1D - 1) / BlockSize1D;
     int gridSize1D_w1 = (INPUT_SIZE * HIDDEN_SIZE + BlockSize1D - 1) / BlockSize1D;
     int gridSize1D_w2 = (HIDDEN_SIZE * OUTPUT_SIZE + BlockSize1D - 1) / BlockSize1D;
     
     // Training loop
     for (int Epoch = 0; Epoch < NUM_EPOCHS; Epoch++) {
         float TotalLoss = 0.0f;
         
         // Process mini-batches
         int NumBatches = MNIST_TRAIN_SIZE / BATCH_SIZE;
         for (int Batch = 0; Batch < NumBatches; Batch++) {
             // Copy batch data to device
             CUDA_CHECK(cudaMemcpy(DImages, 
                                  &TrainImages[Batch * BATCH_SIZE * INPUT_SIZE], 
                                  BATCH_SIZE * INPUT_SIZE * sizeof(float), 
                                  cudaMemcpyHostToDevice));
             CUDA_CHECK(cudaMemcpy(DLabels, 
                                  &TrainLabels[Batch * BATCH_SIZE], 
                                  BATCH_SIZE * sizeof(unsigned char), 
                                  cudaMemcpyHostToDevice));
             
             // Initialize loss to 0
             float Zero = 0.0f;
             CUDA_CHECK(cudaMemcpy(DLoss, &Zero, sizeof(float), cudaMemcpyHostToDevice));
             
             // Forward pass
             
             // 1. Hidden layer: input -> hidden
             MatrixMultiply<<<gridSize2D_hidden, BlockSize2D>>>(
                 DImages, DWeights1, DHiddenPreact, 
                 BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
             
             // Add biases to hidden layer preactivation
             AddBiases<<<gridSize1D_hidden, BlockSize1D>>>(
                 DHiddenPreact, DBiases1, BATCH_SIZE, HIDDEN_SIZE);
             
             // Apply ReLU activation
             relu_activation<<<gridSize1D_hidden, BlockSize1D>>>(
                 DHiddenPreact, DHiddenOutput, BATCH_SIZE * HIDDEN_SIZE);
             
             // 2. Output layer: hidden -> output
             MatrixMultiply<<<gridSize2D_output, BlockSize2D>>>(
                 DHiddenOutput, DWeights2, DOutputPreact, 
                 BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
             
             // Add biases to output layer preactivation
             AddBiases<<<gridSize1D_output, BlockSize1D>>>(
                 DOutputPreact, DBiases2, BATCH_SIZE, OUTPUT_SIZE);
             
             // Apply softmax activation
             SoftmaxActivation<<<gridSize1D_batch, BlockSize1D>>>(
                 DOutputPreact, DOutput, BATCH_SIZE, OUTPUT_SIZE);
             
             // Calculate loss
             CrossEntropyLoss<<<gridSize1D_batch, BlockSize1D>>>(
                 DOutput, DLabels, DLoss, BATCH_SIZE, OUTPUT_SIZE);
             
             // Backward pass
             
             // 1. Output layer error
             OutputLayerError<<<gridSize1D_output, BlockSize1D>>>(
                 DOutput, DLabels, DOutputError, BATCH_SIZE, OUTPUT_SIZE);
             
             // 2. Calculate ReLU derivative for hidden layer
             ReluDerivative<<<gridSize1D_hidden, BlockSize1D>>>(
                 DHiddenPreact, DHiddenDeriv, BATCH_SIZE * HIDDEN_SIZE);
             
             // 3. Hidden layer error
             HiddenLayerError<<<gridSize1D_hidden, BlockSize1D>>>(
                 DOutputError, DWeights2, DHiddenDeriv, DHiddenError,
                 BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
             
             // Update weights and biases
             
             // 1. Update output layer weights
             UpdateWeights<<<gridSize1D_w2, BlockSize1D>>>(
                 DWeights2, DHiddenOutput, DOutputError,
                 LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);
             
             // 2. Update hidden layer weights
             UpdateWeights<<<gridSize1D_w1, BlockSize1D>>>(
                 DWeights1, DImages, DHiddenError,
                 LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE);
             
             // 3. Update output layer biases
             UpdateBiases<<<(OUTPUT_SIZE + BlockSize1D - 1) / BlockSize1D, BlockSize1D>>>(
                 DBiases2, DOutputError, LEARNING_RATE, OUTPUT_SIZE, BATCH_SIZE);
             
             // 4. Update hidden layer biases
             UpdateBiases<<<(HIDDEN_SIZE + BlockSize1D - 1) / BlockSize1D, BlockSize1D>>>(
                 DBiases1, DHiddenError, LEARNING_RATE, HIDDEN_SIZE, BATCH_SIZE);
             
             // Copy loss back to host
             float BatchLoss;
             CUDA_CHECK(cudaMemcpy(&BatchLoss, DLoss, sizeof(float), cudaMemcpyDeviceToHost));
             BatchLoss /= BATCH_SIZE;  // Average loss per sample
             TotalLoss += BatchLoss;
             
             // Check for CUDA errors
             cudaError_t Error = cudaGetLastError();
             if (Error != cudaSuccess) {
                 fprintf(stderr, "CUDA error in batch %d: %s\n", Batch, cudaGetErrorString(Error));
                 exit(EXIT_FAILURE);
             }
         }
         
         // Print epoch results
         printf("Epoch %d/%d: Average Loss = %.4f\n", 
                Epoch + 1, NUM_EPOCHS, TotalLoss / NumBatches);
     }
     
     // Copy updated parameters back to host
     CUDA_CHECK(cudaMemcpy(Weights1, DWeights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
     CUDA_CHECK(cudaMemcpy(Weights2, DWeights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
     CUDA_CHECK(cudaMemcpy(Biases1, DBiases1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
     CUDA_CHECK(cudaMemcpy(Biases2, DBiases2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
     
     // Free device memory
     CUDA_CHECK(cudaFree(DWeights1));
     CUDA_CHECK(cudaFree(DWeights2));
     CUDA_CHECK(cudaFree(DBiases1));
     CUDA_CHECK(cudaFree(DBiases2));
     CUDA_CHECK(cudaFree(DImages));
     CUDA_CHECK(cudaFree(DLabels));
     CUDA_CHECK(cudaFree(DHiddenPreact));
     CUDA_CHECK(cudaFree(DHiddenOutput));
     CUDA_CHECK(cudaFree(DOutputPreact));
     CUDA_CHECK(cudaFree(DOutput));
     CUDA_CHECK(cudaFree(DOutputError));
     CUDA_CHECK(cudaFree(DHiddenError));
     CUDA_CHECK(cudaFree(DHiddenDeriv));
     CUDA_CHECK(cudaFree(DLoss));
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
 void TestNetwork(float* TestImages, unsigned char* TestLabels, 
                 float* Weights1, float* Weights2, float* Biases1, float* Biases2) {
     // Allocate device memory
     float *DImages, *DWeights1, *DWeights2, *DBiases1, *DBiases2;
     unsigned char *DLabels;
     float *DHiddenPreact, *DHiddenOutput, *DOutputPreact, *DOutput;
     
     // Allocate memory for network parameters
     CUDA_CHECK(cudaMalloc(&DWeights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DWeights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DBiases1, HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DBiases2, OUTPUT_SIZE * sizeof(float)));
     
     // Allocate memory for batch data
     CUDA_CHECK(cudaMalloc(&DImages, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DLabels, BATCH_SIZE * sizeof(unsigned char)));
     
     // Allocate memory for intermediate values
     CUDA_CHECK(cudaMalloc(&DHiddenPreact, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DHiddenOutput, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DOutputPreact, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     CUDA_CHECK(cudaMalloc(&DOutput, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
     
     // Copy network parameters to device
     CUDA_CHECK(cudaMemcpy(DWeights1, Weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(DWeights2, Weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(DBiases1, Biases1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(DBiases2, Biases2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
     
     // Define grid and block dimensions for different kernels
     dim3 BlockSize2D(16, 16);
     dim3 gridSize2D_hidden((HIDDEN_SIZE + BlockSize2D.x - 1) / BlockSize2D.x, 
                          (BATCH_SIZE + BlockSize2D.y - 1) / BlockSize2D.y);
     dim3 gridSize2D_output((OUTPUT_SIZE + BlockSize2D.x - 1) / BlockSize2D.x, 
                           (BATCH_SIZE + BlockSize2D.y - 1) / BlockSize2D.y);
     
     int BlockSize1D = 256;
     int gridSize1D_hidden = (BATCH_SIZE * HIDDEN_SIZE + BlockSize1D - 1) / BlockSize1D;
     int gridSize1D_output = (BATCH_SIZE * OUTPUT_SIZE + BlockSize1D - 1) / BlockSize1D;
     int gridSize1D_batch = (BATCH_SIZE + BlockSize1D - 1) / BlockSize1D;
     
     // Test variables
     int TotalCorrect = 0;
     float* HOutput = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
     
     if (HOutput == NULL) {
         fprintf(stderr, "Error: Memory allocation failed for h_output\n");
         exit(EXIT_FAILURE);
     }
     
     // Create confusion matrix
     int ConfusionMatrix[OUTPUT_SIZE][OUTPUT_SIZE] = {0};
     
     // Process mini-batches
     int NumBatches = MNIST_TEST_SIZE / BATCH_SIZE;
     for (int Batch = 0; Batch < NumBatches; Batch++) {
         // Copy batch data to device
         CUDA_CHECK(cudaMemcpy(DImages, 
                              &TestImages[Batch * BATCH_SIZE * INPUT_SIZE], 
                              BATCH_SIZE * INPUT_SIZE * sizeof(float), 
                              cudaMemcpyHostToDevice));
         CUDA_CHECK(cudaMemcpy(DLabels, 
                              &TestLabels[Batch * BATCH_SIZE], 
                              BATCH_SIZE * sizeof(unsigned char), 
                              cudaMemcpyHostToDevice));
         
         // Forward pass
         
         // 1. Hidden layer: input -> hidden
         MatrixMultiply<<<gridSize2D_hidden, BlockSize2D>>>(
             DImages, DWeights1, DHiddenPreact, 
             BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
         
         // Add biases to hidden layer preactivation
         AddBiases<<<gridSize1D_hidden, BlockSize1D>>>(
             DHiddenPreact, DBiases1, BATCH_SIZE, HIDDEN_SIZE);
         
         // Apply ReLU activation
         relu_activation<<<gridSize1D_hidden, BlockSize1D>>>(
             DHiddenPreact, DHiddenOutput, BATCH_SIZE * HIDDEN_SIZE);
         
         // 2. Output layer: hidden -> output
         MatrixMultiply<<<gridSize2D_output, BlockSize2D>>>(
             DHiddenOutput, DWeights2, DOutputPreact, 
             BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
         
         // Add biases to output layer preactivation
         AddBiases<<<gridSize1D_output, BlockSize1D>>>(
             DOutputPreact, DBiases2, BATCH_SIZE, OUTPUT_SIZE);
         
         // Apply softmax activation
         SoftmaxActivation<<<gridSize1D_batch, BlockSize1D>>>(
             DOutputPreact, DOutput, BATCH_SIZE, OUTPUT_SIZE);
         
         // Copy output back to host
         CUDA_CHECK(cudaMemcpy(HOutput, DOutput, 
                              BATCH_SIZE * OUTPUT_SIZE * sizeof(float), 
                              cudaMemcpyDeviceToHost));
         
         // Count correct predictions and update confusion matrix
         for (int i = 0; i < BATCH_SIZE; i++) {
             // Find predicted class (maximum probability)
             int PredictedClass = 0;
             float MaxProb = HOutput[i * OUTPUT_SIZE];
             
             for (int j = 1; j < OUTPUT_SIZE; j++) {
                 if (HOutput[i * OUTPUT_SIZE + j] > MaxProb) {
                     MaxProb = HOutput[i * OUTPUT_SIZE + j];
                     PredictedClass = j;
                 }
             }
             
             // Get true label
             int TrueLabel = TestLabels[Batch * BATCH_SIZE + i];
             
             // Update confusion matrix
             ConfusionMatrix[TrueLabel][PredictedClass]++;
             
             // Check if prediction is correct
             if (PredictedClass == TrueLabel) {
                 TotalCorrect++;
             }
         }
     }
     
     // Print test accuracy
     float Accuracy = (float)TotalCorrect / MNIST_TEST_SIZE * 100.0f;
     printf("Test Accuracy: %.2f%% (%d/%d)\n", 
            Accuracy, TotalCorrect, MNIST_TEST_SIZE);
     
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
             printf("%4d ", ConfusionMatrix[i][j]);
         }
         printf("\n");
     }
     
     // Print per-class accuracy
     printf("\nPer-class Accuracy:\n");
     for (int i = 0; i < OUTPUT_SIZE; i++) {
         int ClassTotal = 0;
         for (int j = 0; j < OUTPUT_SIZE; j++) {
             ClassTotal += ConfusionMatrix[i][j];
         }
         float ClassAccuracy = (float)ConfusionMatrix[i][i] / ClassTotal * 100.0f;
         printf("Class %d: %.2f%%\n", i, ClassAccuracy);
     }
     
     // Free memory
     free(HOutput);
     CUDA_CHECK(cudaFree(DWeights1));
     CUDA_CHECK(cudaFree(DWeights2));
     CUDA_CHECK(cudaFree(DBiases1));
     CUDA_CHECK(cudaFree(DBiases2));
     CUDA_CHECK(cudaFree(DImages));
     CUDA_CHECK(cudaFree(DLabels));
     CUDA_CHECK(cudaFree(DHiddenPreact));
     CUDA_CHECK(cudaFree(DHiddenOutput));
     CUDA_CHECK(cudaFree(DOutputPreact));
     CUDA_CHECK(cudaFree(DOutput));
 }
 
 /**
  * Main function to run the MLP for MNIST classification
  */
 int main() {
     // Set random seed for reproducibility
     srand(time(NULL));
     
     // Define file paths for MNIST dataset
     const char* TrainImagesFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-images.idx3-ubyte";
     const char* TrainLabelsFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/train-labels.idx1-ubyte";
     const char* TestImagesFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-images.idx3-ubyte";
     const char* TestLabelsFile = "/home/jiwokim/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1/t10k-labels.idx1-ubyte";
     
     // Allocate memory for MNIST data
     float* TrainImages = (float*)malloc(MNIST_TRAIN_SIZE * INPUT_SIZE * sizeof(float));
     unsigned char* TrainLabels = (unsigned char*)malloc(MNIST_TRAIN_SIZE * sizeof(unsigned char));
     float* TestImages = (float*)malloc(MNIST_TEST_SIZE * INPUT_SIZE * sizeof(float));
     unsigned char* TestLabels = (unsigned char*)malloc(MNIST_TEST_SIZE * sizeof(unsigned char));
     
     // Check memory allocation
     if (!TrainImages || !TrainLabels || !TestImages || !TestLabels) {
         fprintf(stderr, "Error: Memory allocation failed for dataset\n");
         // Free any successfully allocated memory
         if (TrainImages) free(TrainImages);
         if (TrainLabels) free(TrainLabels);
         if (TestImages) free(TestImages);
         if (TestLabels) free(TestLabels);
         return EXIT_FAILURE;
     }
     
     // Load MNIST data
     printf("Loading MNIST data...\n");
     LoadMNISTData(TrainImagesFile, TrainLabelsFile, 
                  TrainImages, TrainLabels, MNIST_TRAIN_SIZE);
     LoadMNISTData(TestImagesFile, TestLabelsFile, 
                  TestImages, TestLabels, MNIST_TEST_SIZE);
     
     // Allocate memory for network parameters
     float* Weights1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
     float* Weights2 = (float*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
     float* Biases1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
     float* Biases2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
     
     // Check memory allocation
     if (!Weights1 || !Weights2 || !Biases1 || !Biases2) {
         fprintf(stderr, "Error: Memory allocation failed for network parameters\n");
         // Free any successfully allocated memory
         if (TrainImages) free(TrainImages);
         if (TrainLabels) free(TrainLabels);
         if (TestImages) free(TestImages);
         if (TestLabels) free(TestLabels);
         if (Weights1) free(Weights1);
         if (Weights2) free(Weights2);
         if (Biases1) free(Biases1);
         if (Biases2) free(Biases2);
         return EXIT_FAILURE;
     }
     
     // Initialize network parameters
     printf("Initializing network parameters...\n");
     InitializeWeights(Weights1, INPUT_SIZE, HIDDEN_SIZE);
     InitializeWeights(Weights2, HIDDEN_SIZE, OUTPUT_SIZE);
     InitializeBiases(Biases1, HIDDEN_SIZE);
     InitializeBiases(Biases2, OUTPUT_SIZE);
     
     // Print initial network summary
     PrintLayerInfo(Weights1, INPUT_SIZE, HIDDEN_SIZE, "Hidden Layer");
     PrintLayerInfo(Weights2, HIDDEN_SIZE, OUTPUT_SIZE, "Output Layer");
     
     printf("Training MLP for MNIST digit classification...\n");
     
     // Create CUDA events for timing
     cudaEvent_t Start, Stop;
     CUDA_CHECK(cudaEventCreate(&Start));
     CUDA_CHECK(cudaEventCreate(&Stop));
     
     // Record start time
     CUDA_CHECK(cudaEventRecord(Start, 0));
     
     // Train network
     TrainNetwork(TrainImages, TrainLabels, Weights1, Weights2, Biases1, Biases2);
     
     // Record stop time
     CUDA_CHECK(cudaEventRecord(Stop, 0));
     CUDA_CHECK(cudaEventSynchronize(Stop));
     
     // Calculate training time
     float TrainingTime;
     CUDA_CHECK(cudaEventElapsedTime(&TrainingTime, Start, Stop));
     printf("Training completed in %.2f seconds\n", TrainingTime / 1000.0f);
     
     // Print trained network summary
     PrintLayerInfo(Weights1, INPUT_SIZE, HIDDEN_SIZE, "Trained Hidden Layer");
     PrintLayerInfo(Weights2, HIDDEN_SIZE, OUTPUT_SIZE, "Trained Output Layer");
     
     printf("Testing network...\n");
     
     // Test network
     TestNetwork(TestImages, TestLabels, Weights1, Weights2, Biases1, Biases2);
     
     // Free memory
     free(TrainImages);
     free(TrainLabels);
     free(TestImages);
     free(TestLabels);
     free(Weights1);
     free(Weights2);
     free(Biases1);
     free(Biases2);
     CUDA_CHECK(cudaEventDestroy(Start));
     CUDA_CHECK(cudaEventDestroy(Stop));
     
     // Print memory usage and CUDA device info
     size_t FreeMemory, TotalMemory;
     CUDA_CHECK(cudaMemGetInfo(&FreeMemory, &TotalMemory));
     printf("\nCUDA Memory: %.2f MB free / %.2f MB total\n", 
            FreeMemory / (1024.0f * 1024.0f), 
            TotalMemory / (1024.0f * 1024.0f));
     
     cudaDeviceProp DeviceProp;
     CUDA_CHECK(cudaGetDeviceProperties(&DeviceProp, 0));
     printf("GPU: %s\n", DeviceProp.name);
     
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