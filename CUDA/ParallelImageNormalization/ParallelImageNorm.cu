/**
 * @file ParallelImageNorm.cu
 * @brief CUDA parallel image normalization kernel implementations.
 *
 * Normalizes an image by computing per-channel statistics on the GPU:
 *   normalized[p][c] = (pixel[p][c] - mean[c]) / std[c]
 *
 * Complexity:
 *   - O(N * C) total work for N pixels and C channels
 *   - O(log(blockDim)) parallel steps per reduction kernel
 */

#include "ParallelImageNorm.cuh"
#include <math.h>

__global__ void ConvertUcharToFloat(unsigned char* dImageUchar, float* dImageFloat, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        dImageFloat[idx] = (float)dImageUchar[idx];
    }
}

__global__ void SumReductionKernel(float* dImageFloat, float* dPartialSums, int n, int c) {
    extern __shared__ float sSums[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int pixelsPerThread = (n + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    int startPixel = (bid * blockDim.x + tid) * pixelsPerThread;
    int endPixel = min(startPixel + pixelsPerThread, n);

    float localSums[3] = {0.0f, 0.0f, 0.0f};

    for (int p = startPixel; p < endPixel; p++) {
        for (int ch = 0; ch < c; ch++) {
            localSums[ch] += dImageFloat[p * c + ch];
        }
    }

    for (int ch = 0; ch < c; ch++) {
        sSums[ch * blockDim.x + tid] = localSums[ch];
    }
    __syncthreads();

    for (int ch = 0; ch < c; ch++) {
        float* sSumC = sSums + ch * blockDim.x;
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sSumC[tid] += sSumC[tid + s];
            __syncthreads();
        }
        if (tid == 0) dPartialSums[bid * c + ch] = sSumC[0];
    }
}

__global__ void SumOfSquaresReductionKernel(float* dImageFloat, float* dMean,
                                            float* dPartialSums, int n, int c) {
    extern __shared__ float sSums[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int pixelsPerThread = (n + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    int startPixel = (bid * blockDim.x + tid) * pixelsPerThread;
    int endPixel = min(startPixel + pixelsPerThread, n);

    float localSums[3] = {0.0f, 0.0f, 0.0f};

    for (int p = startPixel; p < endPixel; p++) {
        for (int ch = 0; ch < c; ch++) {
            float val = dImageFloat[p * c + ch] - dMean[ch];
            localSums[ch] += val * val;
        }
    }

    for (int ch = 0; ch < c; ch++) {
        sSums[ch * blockDim.x + tid] = localSums[ch];
    }
    __syncthreads();

    for (int ch = 0; ch < c; ch++) {
        float* sSumC = sSums + ch * blockDim.x;
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sSumC[tid] += sSumC[tid + s];
            __syncthreads();
        }
        if (tid == 0) dPartialSums[bid * c + ch] = sSumC[0];
    }
}

__global__ void NormalizeKernel(float* dImageFloat, float* dMean, float* dStddev,
                                float* dNormalized, int size, int c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int channel = idx % c;
        dNormalized[idx] = (dImageFloat[idx] - dMean[channel]) / (dStddev[channel] + 1e-5f);
    }
}
