/**
 * @file main.cu
 * @brief Driver for the CUDA thread indexing demonstration.
 *
 * Launches a kernel with 2 blocks of 30 threads each to show how
 * global thread IDs are computed.
 */

#include "ThreadIdx.cuh"
#include <stdio.h>

int main() {
    const int NUM_THREADS = 30;
    const int NUM_BLOCKS  = 2;

    printf("Launching PrintThreadIdx: %d blocks, %d threads/block\n",
           NUM_BLOCKS, NUM_THREADS);

    PrintThreadIdx<<<NUM_BLOCKS, NUM_THREADS>>>();
    cudaDeviceSynchronize();

    return 0;
}
