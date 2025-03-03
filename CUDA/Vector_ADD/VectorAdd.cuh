#ifndef VECTOR_ADD_CUH
#define VECTOR_ADD_CUH

/**
 * @file VectorAdd.cuh
 * @brief Header for CUDA vector addition kernel.
 *
 * Element-wise vector addition: result[i] = vectorA[i] + vectorB[i].
 * The "Hello World" of GPU programming.
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
 * @brief Element-wise vector addition kernel.
 *
 * Each thread computes exactly one element using its global thread index.
 *
 * @param vectorA  Device pointer to the first input vector.
 * @param vectorB  Device pointer to the second input vector.
 * @param result   Device pointer to the output vector.
 * @param length   Number of elements in each vector.
 */
__global__ void VectorAddKernel(int* vectorA, int* vectorB, int* result, int length);

/**
 * @brief Host API: performs vector addition on the GPU.
 *
 * Allocates device memory, copies data, launches kernel, and copies result back.
 *
 * @param hostA    Host pointer to the first input vector.
 * @param hostB    Host pointer to the second input vector.
 * @param hostC    Host pointer to the output vector.
 * @param length   Number of elements in each vector.
 */
void VectorAdd(const int* hostA, const int* hostB, int* hostC, int length);

#endif // VECTOR_ADD_CUH
