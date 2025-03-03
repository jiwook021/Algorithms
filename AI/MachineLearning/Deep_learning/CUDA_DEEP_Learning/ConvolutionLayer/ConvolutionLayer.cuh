#ifndef CONVOLUTION_LAYER_CUH
#define CONVOLUTION_LAYER_CUH

/**
 * @file ConvolutionLayer.cuh
 * @brief Header for CUDA 2D convolution kernel.
 */

#include <cuda_runtime.h>

#define CUDA_CHECK(Call)                                                      \
    if ((Call) != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(Call));                 \
        exit(EXIT_FAILURE);                                                   \
    }

/**
 * @brief CUDA kernel for 2D convolution (valid padding).
 *
 * @param input       Device pointer to input feature map.
 * @param kernel      Device pointer to convolution kernel.
 * @param output      Device pointer to output feature map.
 * @param inputWidth  Width of input.
 * @param inputHeight Height of input.
 * @param kernelWidth Width of kernel.
 * @param kernelHeight Height of kernel.
 */
__global__ void Conv2DKernel(const float* input, const float* kernel, float* output,
                             int inputWidth, int inputHeight,
                             int kernelWidth, int kernelHeight);

#endif // CONVOLUTION_LAYER_CUH
