/**
 * @file DeadLock.cu
 * @brief CUDA deadlock demonstration kernel implementation.
 *
 * Illustrates warp-level synchronization hazards when __syncthreads() is
 * placed inside divergent branches and spin-locks are used across warps.
 *
 * Reference: "Programming in Parallel with CUDA" by Richard Ansorge (Example 3.8).
 *            Licensed under CC BY-NC 4.0 for non-commercial use.
 */

#include "DeadLock.cuh"
#include <stdio.h>

__global__ void DeadLockKernel(int gsync, int doLock)
{
    __shared__ int lock;
    if (threadIdx.x == 0) lock = 0;
    __syncthreads();  // Normal sync after setting shared memory

    if (threadIdx.x < gsync) {            // Group A
        __syncthreads();                  // Sync A
        if (threadIdx.x == 0) lock = 1;   // Deadlock unless we reach this
    }
    else if (threadIdx.x < 2 * gsync) {   // Group B
        __syncthreads();                  // Sync B
    }

    if (doLock) while (lock != 1);        // Group C may cause deadlock here
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("deadlock OK\n");
}
