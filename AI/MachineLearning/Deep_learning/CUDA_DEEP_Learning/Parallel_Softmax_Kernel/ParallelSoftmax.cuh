#ifndef PARALLEL_SOFTMAX_DL_CUH
#define PARALLEL_SOFTMAX_DL_CUH

/**
 * @file ParallelSoftmax.cuh
 * @brief Header for CUDA softmax + argmax kernels for per-pixel classification.
 */

#include <cuda_runtime.h>

#define CUDA_CHECK(Call)                                                      \
    do {                                                                      \
        cudaError_t Err = Call;                                               \
        if (Err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(Err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void SoftmaxKernel(float* input, float* output, int h, int w, int c);
__global__ void ArgmaxKernel(float* softmaxOutput, int* argmaxOutput, int h, int w, int c);

#endif // PARALLEL_SOFTMAX_DL_CUH
