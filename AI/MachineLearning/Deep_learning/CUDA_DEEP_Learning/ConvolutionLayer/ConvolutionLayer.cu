/**
 * @file ConvolutionLayer.cu
 * @brief CUDA 2D convolution kernel implementation.
 *
 * Performs valid-mode convolution where each thread computes one output element.
 *
 * Complexity:
 *   - Time:  O(kernelW * kernelH) per thread
 *   - Space: O(outputW * outputH) output on device
 */

#include "ConvolutionLayer.cuh"

__global__ void Conv2DKernel(const float* input, const float* kernel, float* output,
                             int inputWidth, int inputHeight,
                             int kernelWidth, int kernelHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outputWidth  = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    if (x >= outputWidth || y >= outputHeight) return;

    float sum = 0.0f;
    for (int ky = 0; ky < kernelHeight; ky++) {
        for (int kx = 0; kx < kernelWidth; kx++) {
            sum += input[(y + ky) * inputWidth + (x + kx)] * kernel[ky * kernelWidth + kx];
        }
    }
    output[y * outputWidth + x] = sum;
}
