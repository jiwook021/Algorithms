#ifndef SUM_MIN_MAX_CUH
#define SUM_MIN_MAX_CUH

/**
 * @file SumMinMax.cuh
 * @brief Header for CUDA parallel reduction kernels: sum, min, and max.
 */

#include <cuda_runtime.h>
#include <float.h>

#define CHECK_CUDA(Call)                                                      \
    do {                                                                      \
        cudaError_t Err = Call;                                               \
        if (Err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(Err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/**
 * @brief Shared-memory sum reduction kernel.
 */
__global__ void ReduceSumKernel(const float* input, float* output, unsigned int n);

/**
 * @brief Shared-memory min reduction kernel.
 */
__global__ void ReduceMinKernel(const float* input, float* output, unsigned int n);

/**
 * @brief Shared-memory max reduction kernel.
 */
__global__ void ReduceMaxKernel(const float* input, float* output, unsigned int n);

#endif // SUM_MIN_MAX_CUH
