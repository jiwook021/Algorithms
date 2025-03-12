#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>  // For FLT_MAX instead of std::numeric_limits

// Error checking utility
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Basic sum reduction kernel
__global__ void reduceSum(const float* input, float* output, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = threadIdx.x;
    
    // Load into shared memory
    sdata[i] = (tid < n) ? input[tid] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) {
            sdata[i] += sdata[i + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (i == 0) output[blockIdx.x] = sdata[0];
}

// Min reduction kernel
__global__ void reduceMin(const float* input, float* output, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = threadIdx.x;
    
    // Load into shared memory
    sdata[i] = (tid < n) ? input[tid] : FLT_MAX;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) {
            sdata[i] = fminf(sdata[i], sdata[i + s]);  // Use fminf
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (i == 0) output[blockIdx.x] = sdata[0];
}

// Max reduction kernel
__global__ void reduceMax(const float* input, float* output, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = threadIdx.x;
    
    // Load into shared memory
    sdata[i] = (tid < n) ? input[tid] : -FLT_MAX;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) {
            sdata[i] = fmaxf(sdata[i], sdata[i + s]);  // Use fmaxf
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (i == 0) output[blockIdx.x] = sdata[0];
}

int main(int argc, char** argv) {
    const int n = 1 << 20;  // 1M elements
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(sizeof(float));
    
    // Initialize array
    for (int i = 0; i < n; i++) {
        h_in[i] = (float)(rand() % 100) / 100.0f;
    }
    
    // Allocate device memory
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float) / 256)); // Assuming block size of 256
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    
    // Setup execution parameters
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch sum kernel
    printf("Running sum reduction...\n");
    reduceSum<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, n);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy result back to host
    float* h_sum = (float*)malloc(gridSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_sum, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute final sum on CPU
    float finalSum = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        finalSum += h_sum[i];
    }
    printf("Sum: %f\n", finalSum);
    
    // Launch min kernel
    printf("Running min reduction...\n");
    reduceMin<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, n);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy result back to host
    float* h_min = (float*)malloc(gridSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_min, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute final min on CPU
    float finalMin = FLT_MAX;
    for (int i = 0; i < gridSize; i++) {
        if (h_min[i] < finalMin) finalMin = h_min[i];
    }
    printf("Min: %f\n", finalMin);
    
    // Launch max kernel
    printf("Running max reduction...\n");
    reduceMax<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, n);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy result back to host
    float* h_max = (float*)malloc(gridSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_max, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute final max on CPU
    float finalMax = -FLT_MAX;
    for (int i = 0; i < gridSize; i++) {
        if (h_max[i] > finalMax) finalMax = h_max[i];
    }
    printf("Max: %f\n", finalMax);
    
    // Free memory
    free(h_in);
    free(h_out);
    free(h_sum);
    free(h_min);
    free(h_max);
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}