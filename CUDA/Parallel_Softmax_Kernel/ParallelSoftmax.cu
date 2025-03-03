/**
 * @file ParallelSoftmax.cu
 * @brief CUDA softmax + argmax kernel implementations.
 *
 * Numerically stable softmax:
 *   softmax[p, k] = exp(input[p, k] - max) / sum(exp(input[p, j] - max))
 *
 * Complexity per pixel: O(C) for C channels.
 */

#include "ParallelSoftmax.cuh"

__global__ void SoftmaxKernel(float* input, float* output, int h, int w, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = h * w;

    if (idx < totalPixels) {
        int base = idx * c;

        // Find max for numerical stability
        float maxVal = input[base];
        for (int k = 1; k < c; k++) {
            float val = input[base + k];
            if (val > maxVal) maxVal = val;
        }

        // Compute sum of exponentials
        float sum = 0.0f;
        for (int k = 0; k < c; k++) {
            sum += expf(input[base + k] - maxVal);
        }

        // Compute softmax
        for (int k = 0; k < c; k++) {
            output[base + k] = expf(input[base + k] - maxVal) / sum;
        }
    }
}

__global__ void ArgmaxKernel(float* softmaxOutput, int* argmaxOutput, int h, int w, int c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = h * w;

    if (idx < totalPixels) {
        int base = idx * c;
        int maxIdx = 0;
        float maxVal = softmaxOutput[base];
        for (int k = 1; k < c; k++) {
            float val = softmaxOutput[base + k];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = k;
            }
        }
        argmaxOutput[idx] = maxIdx;
    }
}
