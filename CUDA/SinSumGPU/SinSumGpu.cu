/**
 * @file SinSumGpu.cu
 * @brief GPU-accelerated sine computation and reduction kernels.
 */

#include "SinSumGpu.cuh"

__global__ void GpuSinKernel(float* sums, int steps, int terms, float stepSize) {
    int step = blockIdx.x * blockDim.x + threadIdx.x;
    if (step < steps) {
        float x = stepSize * step;
        sums[step] = SinSum(x, terms);
    }
}

__global__ void SumArrayKernel(float* arr, int size, float* result) {
    extern __shared__ float smem[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    smem[tid] = (idx < size) ? arr[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, smem[0]);
    }
}
