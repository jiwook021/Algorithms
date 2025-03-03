#ifndef MATRIX_MULTIPLY_CUH
#define MATRIX_MULTIPLY_CUH

/**
 * @file MatrixMultiply.cuh
 * @brief Header for CUDA matrix multiplication kernels (naive and tiled).
 */

#include <cuda_runtime.h>

#define CUDA_CHECK(Call)                                                      \
    do {                                                                      \
        cudaError_t Err = Call;                                               \
        if (Err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(Err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define TILE_SIZE 16

/**
 * @brief Naive matrix multiplication kernel. One thread per output element.
 */
__global__ void MatrixMultiplyKernel(const float* A, const float* B, float* C,
                                     int aRows, int aCols, int bCols);

/**
 * @brief Tiled matrix multiplication using shared memory.
 * Reduces global memory accesses by factor of TILE_SIZE.
 */
__global__ void MatrixMultiplyTiledKernel(const float* A, const float* B, float* C,
                                          int aRows, int aCols, int bCols);

/**
 * @brief Host API: performs matrix multiplication C = A * B using CUDA (naive kernel).
 * @return true on success.
 */
bool MatrixMultiplyCuda(const float* A, const float* B, float* C,
                        int aRows, int aCols, int bCols);

/**
 * @brief CPU reference implementation.
 */
void MatrixMultiplyCpu(const float* A, const float* B, float* C,
                       int aRows, int aCols, int bCols);

/**
 * @brief Print a matrix to stdout.
 */
void PrintMatrix(const float* matrix, int rows, int cols);

#endif // MATRIX_MULTIPLY_CUH
