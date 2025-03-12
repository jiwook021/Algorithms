#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

// Function to compute sine using Taylor series
__host__ __device__ inline float sinsum(float x, int terms)
{
    float x2 = x * x;
    float term = x;   // first term of series
    float sum = term; // sum of terms so far
    for(int n = 1; n < terms; n++){
        term *= -x2 / (2*n*(2*n+1));  // build factorial
        sum += term;
    }
    return sum;
}

// CUDA kernel for parallel sine calculation
__global__ void gpu_sin(float *sums, int steps, int terms, float step_size)
{
    int step = blockIdx.x*blockDim.x+threadIdx.x; // unique thread ID
    if(step < steps){
        float x = step_size*step;
        sums[step] = sinsum(x,terms);  // store sin values in array
    }
}

// Simple kernel to calculate sum (for small arrays)
__global__ void sum_array(float *arr, int size, float *result)
{
    extern __shared__ float smem[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    // Initialize shared memory
    if (idx < size)
        smem[tid] = arr[idx];
    else
        smem[tid] = 0;
        
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block
    if (tid == 0) {
        atomicAdd(result, smem[0]);
    }
}

int main(int argc, char *argv[])
{
    // Process command line arguments
    int steps = (argc > 1) ? atoi(argv[1]) : 10000000; // Default 10M steps
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;     // Default 1000 terms
    
    // Configure CUDA execution parameters
    int threads = 256;
    int blocks = (steps + threads - 1) / threads;  // ensure threads*blocks ≥ steps
    
    // Set up calculation parameters
    double pi = 3.14159265358979323;
    double step_size = pi / (steps - 1); // NB n-1 steps between n points
    
    // Allocate device memory
    float *d_sums = NULL;
    cudaError_t cudaStatus = cudaMalloc((void**)&d_sums, steps * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    // Create device result variable for sum
    float *d_sum = NULL;
    cudaStatus = cudaMalloc((void**)&d_sum, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_sum: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_sums);
        return 1;
    }
    
    // Initialize d_sum to 0
    cudaStatus = cudaMemset(d_sum, 0, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_sums);
        cudaFree(d_sum);
        return 1;
    }
    
    // Start timing
    clock_t start = clock();
    
    // Launch kernel to compute sin values
    gpu_sin<<<blocks, threads>>>(d_sums, steps, terms, (float)step_size);
    
    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_sums);
        cudaFree(d_sum);
        return 1;
    }
    
    // Wait for GPU to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_sums);
        cudaFree(d_sum);
        return 1;
    }
    
    // Launch reduction kernel to sum all values
    sum_array<<<blocks, threads, threads * sizeof(float)>>>(d_sums, steps, d_sum);
    
    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Reduction kernel failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_sums);
        cudaFree(d_sum);
        return 1;
    }
    
    // Wait for GPU to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_sums);
        cudaFree(d_sum);
        return 1;
    }
    
    // Copy result back to host
    float h_sum;
    cudaStatus = cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_sums);
        cudaFree(d_sum);
        return 1;
    }
    
    // Trapezoidal Rule Correction
    double gpu_sum = h_sum;
    gpu_sum -= 0.5 * (sinsum(0.0f, terms) + sinsum((float)pi, terms));
    gpu_sum *= step_size;
    
    // End timing
    clock_t end = clock();
    double gpu_time = 1000.0 * (double)(end - start) / CLOCKS_PER_SEC; // Convert to milliseconds
    
    // Print results
    printf("gpu sum = %.10f, steps %d terms %d time %.3f ms\n",
        gpu_sum, steps, terms, gpu_time);
    
    // Clean up
    cudaFree(d_sums);
    cudaFree(d_sum);
    
    return 0;
}