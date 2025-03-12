/**
 * @file histogram.cu
 * @brief CUDA implementation for histogram calculation
 * 
 * This program demonstrates two methods of histogram computation:
 * 1. Using random data generated on the host
 * 2. Processing image-like data (simulated)
 * 
 * The implementation handles race conditions using atomic operations
 * and demonstrates efficient memory management between host and device.
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
  * CUDA kernel for histogram calculation
  * 
  * Uses atomicAdd to safely update bin counts from multiple threads
  * 
  * @param data Input data array
  * @param histogram Output histogram array
  * @param dataSize Size of the input data
  * @param numBins Number of histogram bins
  */
 __global__ void histogramKernel(const unsigned char* data, 
                                 unsigned int* histogram, 
                                 int dataSize, 
                                 int numBins) {
     // Calculate global thread ID
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Stride for grid-stride loop pattern
     int stride = blockDim.x * gridDim.x;
     
     // Each thread processes multiple elements with stride
     for (int i = tid; i < dataSize; i += stride) {
         // Ensure the value is within range
         if (data[i] < numBins) {
             // Atomically increment the appropriate histogram bin
             // This prevents race conditions when multiple threads
             // try to update the same bin simultaneously
             atomicAdd(&histogram[data[i]], 1);
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
  * Generate random data for histogram calculation
  * 
  * @param data Output array for random data
  * @param size Number of data points to generate
  * @param maxValue Maximum value of random data
  */
 void generateRandomData(unsigned char* data, int size, int maxValue) {
     for (int i = 0; i < size; i++) {
         data[i] = rand() % (maxValue + 1);
     }
 }
 
 /**
  * Print histogram data in a compact format
  * 
  * @param histogram Histogram array to print
  * @param numBins Number of bins in the histogram
  */
 void printHistogram(const unsigned int* histogram, int numBins) {
     // Find maximum count for scaling
     unsigned int maxCount = 0;
     for (int i = 0; i < numBins; i++) {
         if (histogram[i] > maxCount) {
             maxCount = histogram[i];
         }
     }
     
     // Print histogram header
     printf("Histogram (showing non-zero bins):\n");
     printf("Bin\tCount\n");
     printf("-------------------\n");
     
     // Print non-zero bins
     for (int i = 0; i < numBins; i++) {
         if (histogram[i] > 0) {
             printf("%3d\t%7u\n", i, histogram[i]);
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
     // Process command line arguments for data size
     int dataSize = DEFAULT_DATA_SIZE;
     if (argc > 1) {
         dataSize = atoi(argv[1]);
         if (dataSize <= 0) {
             fprintf(stderr, "Invalid data size. Using default: %d\n", DEFAULT_DATA_SIZE);
             dataSize = DEFAULT_DATA_SIZE;
         }
     }
     
     printf("Computing histogram for %d data points with %d bins\n", dataSize, NUM_BINS);
     
     // Seed random number generator
     srand(time(NULL));
     
     // Allocate memory for input data on the host
     unsigned char* h_data = NULL;
     CUDA_CHECK(cudaMallocHost(&h_data, dataSize * sizeof(unsigned char)));
     
     // Generate random data
     generateRandomData(h_data, dataSize, MAX_PIXEL_VALUE);
     
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
     
     // Calculate grid size based on data size
     int gridSize = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
     // Cap grid size to avoid excessive overhead
     if (gridSize > 65535) gridSize = 65535;
     
     // Launch kernel with timing
     CUDA_CHECK(cudaEventRecord(start));
     
     histogramKernel<<<gridSize, BLOCK_SIZE>>>(d_data, d_histogram, dataSize, NUM_BINS);
     
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
     
     // Print the histogram (first few bins)
     printf("\nHistogram Preview (first 10 bins):\n");
     printf("Bin\tCount\n");
     printf("----------------\n");
     for (int i = 0; i < 10 && i < NUM_BINS; i++) {
         printf("%3d\t%7u\n", i, h_histogramGPU[i]);
     }
     
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