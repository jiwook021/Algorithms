/**
 * @file ThreadIdx.cu
 * @brief CUDA thread indexing demonstration kernel implementation.
 *
 * Each thread prints its global index, demonstrating the fundamental
 * CUDA thread identification formula.
 */

#include "ThreadIdx.cuh"
#include <stdio.h>

__global__ void PrintThreadIdx() {
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread index: %d\n", globalIndex);
}
