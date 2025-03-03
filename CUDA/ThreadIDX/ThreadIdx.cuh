#ifndef THREAD_IDX_CUH
#define THREAD_IDX_CUH

/**
 * @file ThreadIdx.cuh
 * @brief Header for CUDA thread indexing demonstration kernel.
 *
 * Demonstrates the fundamental global thread ID formula:
 *   globalIndex = blockIdx.x * blockDim.x + threadIdx.x
 */

#include <cuda_runtime.h>

/**
 * @brief Kernel that prints the global thread index for each thread.
 */
__global__ void PrintThreadIdx();

#endif // THREAD_IDX_CUH
