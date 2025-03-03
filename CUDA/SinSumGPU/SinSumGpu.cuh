#ifndef SIN_SUM_GPU_CUH
#define SIN_SUM_GPU_CUH

/**
 * @file SinSumGpu.cuh
 * @brief Header for GPU-accelerated sine computation using Taylor series.
 */

#include <cuda_runtime.h>

/**
 * @brief Compute sin(x) using Taylor series expansion.
 * sin(x) = x - x^3/3! + x^5/5! - ...
 * Marked __host__ __device__ for use on both CPU and GPU.
 */
__host__ __device__ inline float SinSum(float x, int terms) {
    float x2   = x * x;
    float term  = x;
    float sum   = term;
    for (int n = 1; n < terms; n++) {
        term *= -x2 / (2 * n * (2 * n + 1));
        sum  += term;
    }
    return sum;
}

/**
 * @brief Kernel: compute sin(x_i) for N evenly spaced points.
 */
__global__ void GpuSinKernel(float* sums, int steps, int terms, float stepSize);

/**
 * @brief Kernel: shared-memory reduction to sum an array.
 */
__global__ void SumArrayKernel(float* arr, int size, float* result);

#endif // SIN_SUM_GPU_CUH
