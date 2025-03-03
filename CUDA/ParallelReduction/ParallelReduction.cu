/**
 * @file ParallelReduction.cu
 * @brief CUDA parallel reduction (sum) kernel implementation.
 *
 * Algorithm (iterative halving):
 *   Round 1: x[i] += x[i + N/2]  for i in [0, N/2)
 *   Round 2: x[i] += x[i + N/4]  for i in [0, N/4)
 *   ...
 *   Final sum is in x[0].
 *
 * Complexity:
 *   - O(N) total work, O(log N) parallel steps.
 *
 * Note: This is a naive implementation. Production code would use
 * shared memory, warp-level primitives, and loop unrolling.
 */

#include "ParallelReduction.cuh"

__global__ void ReduceKernel(float* x, int m) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    x[tid] += x[tid + m];
}
