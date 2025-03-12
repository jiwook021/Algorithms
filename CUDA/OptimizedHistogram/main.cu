/**
 * @file shared_histogram.cu
 * @brief Optimized CUDA implementation for histogram calculation using shared memory
 * 
 * This program demonstrates an optimized approach to histogram computation
 * using CUDA shared memory to reduce global memory atomic operations,
 * which can significantly improve performance.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <time.h>
 #include <cuda_runtime.h>
 
 // Error checking macro for CUDA operations
 #define CUDA_CHECK(call) \
     do { \
         cudaError_t error = call; \
         if (error != cudaSuccess) { \
             fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(error)); \
             exit(EXIT_FAILURE); \
         } \
     } while(0)
 
 // Configurable parameters
 #define NUM_BINS 256           // Number of histogram bins (for 8-bit values)
 #define MAX_PIXEL_VALUE 255    // Maximum pixel value (8-bit)
 #define BLOCK_SIZE 256         // CUDA threads per block
 #define DEFAULT_DATA_SIZE 10000000 // Default number of data points
 
 /**
  * CUDA kernel for histogram calculation with shared memory optimization
  * 
  * Each thread block maintains a local histogram in shared memory,
  * which is then atomically added to the global histogram.
  * 
  * @param data Input data array
  * @param histogram Output histogram array
  * @param dataSize Size of the input data
  * @param numBins Number of histogram bins
  */
 __global__ void histogramSharedKernel(const unsigned char* data, 
                                      unsigned int* histogram, 
                                      int dataSize, 
                                      int numBins) {
     // Declare shared memory for local histogram
     // This reduces contention on global memory atomics
     __shared__ unsigned int sharedHistogram[NUM_BINS];
     
     // Calculate global thread ID
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     int localTid = threadIdx.x;
     
     // Initialize shared memory histogram bins
     // Each thread initializes one or more bins
     for (int i = localTid; i < numBins; i += blockDim.x) {
         sharedHistogram[i] = 0;
     }
     
     // Ensure all threads have initialized shared memory
     __syncthreads();
     
     // Process data with grid-stride loop
     for (int i = tid; i < dataSize; i += blockDim.x * gridDim.x) {
         if (data[i] < numBins) {
             // Update shared histogram using atomic operation
             // This still requires atomics but only within the block's shared memory
             atomicAdd(&sharedHistogram[data[i]], 1);
         }
     }
     
     // Wait for all threads in block to finish updating shared memory
     __syncthreads();
     
     // Add block's local histogram to global histogram
     // Each thread adds one or more bins
     for (int i = localTid; i < numBins; i += blockDim.x) {
         if (sharedHistogram[i] > 0) {
             atomicAdd(&histogram[i], sharedHistogram[i]);
         }
     }
 }
 
 /**
  * Simple CPU implementation of histogram for verification
  * 
  * @param data Input data array
  * @param histogram Output histogram array
  * @param dataSize Size of the input data
  * @param numBins Number of histogram bins
  */
 void computeHistogramCPU(const unsigned char* data, 
                          unsigned int* histogram, 
                          int dataSize, 
                          int numBins) {
     // Zero out histogram
     memset(histogram, 0, numBins * sizeof(unsigned int));
     
     // Process each data element
     for (int i = 0; i < dataSize; i++) {
         if (data[i] < numBins) {
             histogram[data[i]]++;
         }
     }
 }
 
 /**
  * Generate random data for histogram calculation with controllable distribution
  * 
  * @param data Output array for random data
  * @param size Number of data points to generate
  * @param maxValue Maximum value of random data
  * @param distribution 0=uniform, 1=normal-like, 2=skewed
  */
 void generateRandomData(unsigned char* data, int size, int maxValue, int distribution = 0) {
     switch (distribution) {
         case 1: {
             // Generate normal-like distribution centered around middle value
             int center = maxValue / 2;
             int spread = maxValue / 4;
             for (int i = 0; i < size; i++) {
                 // Sum 3 random values for approximate normal distribution
                 int val = center + (rand() % (2 * spread) - spread + 
                                     rand() % (2 * spread) - spread + 
                                     rand() % (2 * spread) - spread) / 3;
                 // Clamp to valid range
                 data[i] = (val < 0) ? 0 : ((val > maxValue) ? maxValue : val);
             }
             break;
         }
         case 2: {
             // Generate skewed distribution (more lower values)
             for (int i = 0; i < size; i++) {
                 // Square of random value produces skew toward lower values
                 float r = (float)rand() / RAND_MAX;
                 data[i] = (unsigned char)(r * r * maxValue);
             }
             break;
         }
         default: {
             // Uniform distribution
             for (int i = 0; i < size; i++) {
                 data[i] = rand() % (maxValue + 1);
             }
         }
     }
 }
 
 /**
  * Print histogram data in a compact format
  * 
  * @param histogram Histogram array to print
  * @param numBins Number of bins in the histogram
  * @param maxRows Maximum rows to print (0 for all)
  */
 void printHistogram(const unsigned int* histogram, int numBins, int maxRows = 0) {
     // Find maximum count for scaling
     unsigned int maxCount = 0;
     for (int i = 0; i < numBins; i++) {
         if (histogram[i] > maxCount) {
             maxCount = histogram[i];
         }
     }
     
     // Print histogram header
     printf("Histogram %s:\n", maxRows > 0 ? "Preview" : "");
     printf("Bin\tCount\t\tPercentage\n");
     printf("-------------------------------\n");
     
     // Calculate total count for percentage
     unsigned int totalCount = 0;
     for (int i = 0; i < numBins; i++) {
         totalCount += histogram[i];
     }
     
     // Print bins
     int countPrinted = 0;
     for (int i = 0; i < numBins; i++) {
         if (histogram[i] > 0) {
             float percentage = 100.0f * histogram[i] / totalCount;
             printf("%3d\t%7u\t\t%.2f%%\n", i, histogram[i], percentage);
             countPrinted++;
             if (maxRows > 0 && countPrinted >= maxRows) {
                 printf("... (more bins not shown) ...\n");
                 break;
             }
         }
     }
 }
 
 /**
  * Compare CPU and GPU histograms for verification
  * 
  * @param cpuHist CPU-computed histogram
  * @param gpuHist GPU-computed histogram
  * @param numBins Number of bins in the histograms
  * @return true if histograms match, false otherwise
  */
 bool compareHistograms(const unsigned int* cpuHist, 
                        const unsigned int* gpuHist, 
                        int numBins) {
     for (int i = 0; i < numBins; i++) {
         if (cpuHist[i] != gpuHist[i]) {
             printf("Mismatch at bin %d: CPU = %u, GPU = %u\n", 
                    i, cpuHist[i], gpuHist[i]);
             return false;
         }
     }
     return true;
 }
 
 /**
  * Main function to demonstrate histogram calculation
  */
 int main(int argc, char** argv) {
     // Process command line arguments for data size and distribution
     int dataSize = DEFAULT_DATA_SIZE;
     int distribution = 0;
     
     if (argc > 1) {
         dataSize = atoi(argv[1]);
         if (dataSize <= 0) {
             fprintf(stderr, "Invalid data size. Using default: %d\n", DEFAULT_DATA_SIZE);
             dataSize = DEFAULT_DATA_SIZE;
         }
     }
     
     if (argc > 2) {
         distribution = atoi(argv[2]);
         if (distribution < 0 || distribution > 2) {
             fprintf(stderr, "Invalid distribution (0=uniform, 1=normal, 2=skewed). Using uniform.\n");
             distribution = 0;
         }
     }
     
     printf("Computing histogram for %d data points with %d bins\n", dataSize, NUM_BINS);
     printf("Using distribution: %s\n", 
            distribution == 0 ? "Uniform" : 
            (distribution == 1 ? "Normal-like" : "Skewed"));
     
     // Seed random number generator
     srand(time(NULL));
     
     // Allocate memory for input data on the host (pinned memory for better transfer)
     unsigned char* h_data = NULL;
     CUDA_CHECK(cudaMallocHost(&h_data, dataSize * sizeof(unsigned char)));
     
     // Generate random data with specified distribution
     generateRandomData(h_data, dataSize, MAX_PIXEL_VALUE, distribution);
     
     // Allocate memory for the histograms on the host
     unsigned int* h_histogramGPU = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
     unsigned int* h_histogramCPU = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
     
     if (!h_histogramGPU || !h_histogramCPU) {
         fprintf(stderr, "Failed to allocate host memory for histograms\n");
         exit(EXIT_FAILURE);
     }
     
     // Zero out the histograms
     memset(h_histogramGPU, 0, NUM_BINS * sizeof(unsigned int));
     memset(h_histogramCPU, 0, NUM_BINS * sizeof(unsigned int));
     
     // Allocate device memory for data and histogram
     unsigned char* d_data = NULL;
     unsigned int* d_histogram = NULL;
     
     CUDA_CHECK(cudaMalloc(&d_data, dataSize * sizeof(unsigned char)));
     CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BINS * sizeof(unsigned int)));
     
     // Copy data from host to device
     CUDA_CHECK(cudaMemcpy(d_data, h_data, dataSize * sizeof(unsigned char), 
                          cudaMemcpyHostToDevice));
     
     // Initialize device histogram to zeros
     CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(unsigned int)));
     
     // Set up timing for GPU computation
     cudaEvent_t start, stop;
     CUDA_CHECK(cudaEventCreate(&start));
     CUDA_CHECK(cudaEventCreate(&stop));
     
     // Calculate optimal grid size based on data size and GPU properties
     cudaDeviceProp deviceProp;
     CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
     
     int gridSize = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
     // Cap grid size based on device capability
     int maxGridSize = deviceProp.maxGridSize[0];
     if (gridSize > maxGridSize) gridSize = maxGridSize;
     
     printf("\nLaunching kernel with grid size = %d, block size = %d\n", gridSize, BLOCK_SIZE);
     
     // Launch kernel with timing
     CUDA_CHECK(cudaEventRecord(start));
     
     histogramSharedKernel<<<gridSize, BLOCK_SIZE>>>(d_data, d_histogram, dataSize, NUM_BINS);
     
     // Check for kernel launch errors
     CUDA_CHECK(cudaGetLastError());
     CUDA_CHECK(cudaEventRecord(stop));
     
     // Wait for kernel to finish
     CUDA_CHECK(cudaDeviceSynchronize());
     
     // Calculate elapsed time
     float gpuElapsedTime;
     CUDA_CHECK(cudaEventElapsedTime(&gpuElapsedTime, start, stop));
     
     // Copy histogram from device to host
     CUDA_CHECK(cudaMemcpy(h_histogramGPU, d_histogram, NUM_BINS * sizeof(unsigned int), 
                          cudaMemcpyDeviceToHost));
     
     // Compute histogram on CPU for verification
     clock_t cpuStartTime = clock();
     computeHistogramCPU(h_data, h_histogramCPU, dataSize, NUM_BINS);
     clock_t cpuEndTime = clock();
     
     // Calculate CPU elapsed time
     float cpuElapsedTime = 1000.0f * (float)(cpuEndTime - cpuStartTime) / CLOCKS_PER_SEC;
     
     // Verify results
     bool histogramsMatch = compareHistograms(h_histogramCPU, h_histogramGPU, NUM_BINS);
     
     // Print timing information and verification result
     printf("\nPerformance Results:\n");
     printf("GPU time: %.3f ms\n", gpuElapsedTime);
     printf("CPU time: %.3f ms\n", cpuElapsedTime);
     printf("Speedup: %.2fx\n", cpuElapsedTime / gpuElapsedTime);
     printf("Verification: %s\n", histogramsMatch ? "PASSED" : "FAILED");
     
     // Print histogram preview
     printf("\n");
     printHistogram(h_histogramGPU, NUM_BINS, 10);
     
     // Clean up
     CUDA_CHECK(cudaFree(d_data));
     CUDA_CHECK(cudaFree(d_histogram));
     CUDA_CHECK(cudaFreeHost(h_data));
     free(h_histogramGPU);
     free(h_histogramCPU);
     CUDA_CHECK(cudaEventDestroy(start));
     CUDA_CHECK(cudaEventDestroy(stop));
     
     return 0;
 }