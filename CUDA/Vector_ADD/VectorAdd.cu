/**
 * @file VectorAdd.cu
 * @brief CUDA vector addition kernel and host API implementation.
 *
 * Demonstrates the fundamental CUDA workflow: allocate, transfer, compute, transfer back.
 *
 * Complexity:
 *   - Time:  O(1) per thread, O(N) total work
 *   - Space: O(N) on both host and device
 */

#include "VectorAdd.cuh"
#include <cstdio>
#include <cstdlib>

__global__ void VectorAddKernel(int* vectorA, int* vectorB, int* result, int length) {
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIndex < length) {
        result[globalIndex] = vectorA[globalIndex] + vectorB[globalIndex];
    }
}

void VectorAdd(const int* hostA, const int* hostB, int* hostC, int length) {
    int size = length * sizeof(int);

    int *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, size));
    CHECK_CUDA(cudaMalloc(&dB, size));
    CHECK_CUDA(cudaMalloc(&dC, size));

    CHECK_CUDA(cudaMemcpy(dA, hostA, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hostB, size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize  = (length + blockSize - 1) / blockSize;
    VectorAddKernel<<<gridSize, blockSize>>>(dA, dB, dC, length);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hostC, dC, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
}
