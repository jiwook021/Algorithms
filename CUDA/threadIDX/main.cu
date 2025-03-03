#include <stdio.h>
#include <cuda.h>

// CUDA kernel: Each thread prints its threadIdx.x value
__global__ void printThreadIdx() {
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread index: %d\n", globalIndex);
}

int main() {
    const int numThreads = 30;  // Number of threads per block

    // Launch the kernel with 1 block and numThreads threads
    printThreadIdx<<<2, numThreads>>>();

    // Wait for the kernel to finish execution
    cudaDeviceSynchronize();

    return 0;
}