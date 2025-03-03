/**
 * @file SumMinMax.cu
 * @brief CUDA parallel reduction kernels for sum, min, and max.
 *
 * Uses shared-memory tree-reduction with __syncthreads() barriers.
 *
 * Complexity per block:
 *   - O(blockDim) work, O(log blockDim) parallel steps.
 */

#include "SumMinMax.cuh"

__global__ void ReduceSumKernel(const float* input, float* output, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i   = threadIdx.x;

    sdata[i] = (tid < n) ? input[tid] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) sdata[i] += sdata[i + s];
        __syncthreads();
    }

    if (i == 0) output[blockIdx.x] = sdata[0];
}

__global__ void ReduceMinKernel(const float* input, float* output, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i   = threadIdx.x;

    sdata[i] = (tid < n) ? input[tid] : FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) sdata[i] = fminf(sdata[i], sdata[i + s]);
        __syncthreads();
    }

    if (i == 0) output[blockIdx.x] = sdata[0];
}

__global__ void ReduceMaxKernel(const float* input, float* output, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i   = threadIdx.x;

    sdata[i] = (tid < n) ? input[tid] : -FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s) sdata[i] = fmaxf(sdata[i], sdata[i + s]);
        __syncthreads();
    }

    if (i == 0) output[blockIdx.x] = sdata[0];
}
