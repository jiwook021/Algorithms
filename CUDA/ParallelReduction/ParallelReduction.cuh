#ifndef PARALLEL_REDUCTION_CUH
#define PARALLEL_REDUCTION_CUH

/**
 * @file ParallelReduction.cuh
 * @brief Header for CUDA parallel reduction (sum) kernel.
 */

#include <cuda_runtime.h>

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
 * @brief Naive parallel reduction kernel using iterative halving.
 *
 * Each invocation adds x[tid] += x[tid + m], halving the working set.
 *
 * @param x  Device array of floats (modified in-place).
 * @param m  Half the current working set size.
 */
__global__ void ReduceKernel(float* x, int m);

#endif // PARALLEL_REDUCTION_CUH
