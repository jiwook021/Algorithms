#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/**
 * Simple C-style timer function using timespec
 * @return Time in milliseconds
 */
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/**
 * CUDA error checking function
 */
void check_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d - %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(call) check_cuda_error(call, __FILE__, __LINE__)

/**
 * CUDA kernel for parallel reduction
 * @param x Pointer to array of floats in device memory
 * @param m Half the current working set size
 */
__global__ void reduce0(float *x, int m) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    x[tid] += x[tid + m];
}

/**
 * Generate random float between 0 and 1
 */
float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

int main(int argc, char *argv[]) {
    // Parse command line arguments or use default size
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 24; // default 2^24 = 16,777,216 elements
    
    // Allocate host and device memory
    float* h_x = (float*)malloc(N * sizeof(float));
    float* d_x = NULL;
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));

    // Initialize host array with random numbers between 0 and 1
    srand(12345678);  // Fixed seed for reproducibility
    for (int k = 0; k < N; k++) {
        h_x[k] = rand_float();
    }

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Perform CPU reduction and measure time
    double t_start = get_time_ms();
    double host_sum = 0.0;
    for (int k = 0; k < N; k++) {
        host_sum += h_x[k]; // Sequential reduction on host
    }
    double t1 = get_time_ms() - t_start;

    // Perform GPU reduction for N = power of 2
    t_start = get_time_ms();
    for (int m = N/2; m > 0; m /= 2) {
        // Calculate grid dimensions
        int threads = (m < 256) ? m : 256;   // Max 256 threads per block
        int blocks = (m / 256) > 0 ? (m / 256) : 1;  // At least 1 block
        
        // Launch kernel to reduce current level
        reduce0<<<blocks, threads>>>(d_x, m);
        CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
    }
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for all kernels to complete
    double t2 = get_time_ms() - t_start;

    // Get the final result (sum is in the first element)
    float gpu_sum = 0.0f;
    CHECK_CUDA(cudaMemcpy(&gpu_sum, d_x, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results with formatting to match original output
    printf("sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n", 
           N, host_sum, t1, (double)gpu_sum, t2);
    
    // Free allocated memory
    free(h_x);
    CHECK_CUDA(cudaFree(d_x));
    
    return 0;
}