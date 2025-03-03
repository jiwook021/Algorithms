/**
 * @file MaxPooling.cu
 * @brief CUDA max-pooling kernel and host API implementation.
 *
 * Max-pooling reduces spatial dimensions while preserving the most prominent
 * features (maximum activations) in each local region. Each CUDA thread
 * computes exactly one output element.
 *
 * Complexity:
 *   - Time:  O(poolWidth * poolHeight) per thread
 *   - Space: O(inputWidth * inputHeight + outputWidth * outputHeight) on device
 */

#include "MaxPooling.cuh"
#include <cassert>
#include <cfloat>
#include <iostream>

// ---------------------------------------------------------------------------
// CUDA Kernel
// ---------------------------------------------------------------------------

__global__ void MaxPoolingKernel(const float* Input, float* Output,
                                 int InputWidth, int InputHeight,
                                 int PoolWidth, int PoolHeight,
                                 int OutputWidth, int OutputHeight) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outX >= OutputWidth || outY >= OutputHeight)
        return;

    int startX = outX * PoolWidth;
    int startY = outY * PoolHeight;

    float maxVal = -FLT_MAX;

    for (int row = 0; row < PoolHeight; ++row) {
        for (int col = 0; col < PoolWidth; ++col) {
            int curX = startX + col;
            int curY = startY + row;

            if (curX < InputWidth && curY < InputHeight) {
                float val = Input[curY * InputWidth + curX];
                maxVal = fmaxf(maxVal, val);
            }
        }
    }

    Output[outY * OutputWidth + outX] = maxVal;
}

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

void MaxPooling(const std::vector<float>& Input, std::vector<float>& Output,
                int InputWidth, int InputHeight,
                int PoolWidth, int PoolHeight) {

    assert(Input.size() == static_cast<size_t>(InputWidth * InputHeight));

    int OutputWidth  = (InputWidth  + PoolWidth  - 1) / PoolWidth;
    int OutputHeight = (InputHeight + PoolHeight - 1) / PoolHeight;
    Output.resize(OutputWidth * OutputHeight);

    float *dInput  = nullptr;
    float *dOutput = nullptr;

    size_t inputBytes  = Input.size()  * sizeof(float);
    size_t outputBytes = Output.size() * sizeof(float);

    CHECK_CUDA(cudaMalloc(&dInput, inputBytes));
    CHECK_CUDA(cudaMalloc(&dOutput, outputBytes));
    CHECK_CUDA(cudaMemcpy(dInput, Input.data(), inputBytes, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((OutputWidth  + blockSize.x - 1) / blockSize.x,
                  (OutputHeight + blockSize.y - 1) / blockSize.y);

    MaxPoolingKernel<<<gridSize, blockSize>>>(dInput, dOutput,
                                              InputWidth, InputHeight,
                                              PoolWidth, PoolHeight,
                                              OutputWidth, OutputHeight);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(Output.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dInput));
    CHECK_CUDA(cudaFree(dOutput));
}
