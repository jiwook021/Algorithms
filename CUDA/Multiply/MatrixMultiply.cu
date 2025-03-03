/**
 * @file MatrixMultiply.cu
 * @brief CUDA matrix multiplication: naive and tiled (shared memory) kernels.
 *
 * Complexity:
 *   - Naive: O(M * N * K) total, O(K) per thread
 *   - Tiled: Same total work but ~TILE_SIZE fewer global memory accesses
 */

#include "MatrixMultiply.cuh"
#include <iostream>
#include <cmath>

__global__ void MatrixMultiplyKernel(const float* A, const float* B, float* C,
                                     int aRows, int aCols, int bCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < aRows && col < bCols) {
        float sum = 0.0f;
        for (int k = 0; k < aCols; k++) {
            sum += A[row * aCols + k] * B[k * bCols + col];
        }
        C[row * bCols + col] = sum;
    }
}

__global__ void MatrixMultiplyTiledKernel(const float* A, const float* B, float* C,
                                          int aRows, int aCols, int bCols) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (aCols + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < aRows && aCol < aCols)
            ? A[row * aCols + aCol] : 0.0f;

        int bRow = t * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (bRow < aCols && col < bCols)
            ? B[bRow * bCols + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < aRows && col < bCols) {
        C[row * bCols + col] = sum;
    }
}

bool MatrixMultiplyCuda(const float* A, const float* B, float* C,
                        int aRows, int aCols, int bCols) {
    if (!A || !B || !C || aRows <= 0 || aCols <= 0 || bCols <= 0) return false;

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, aRows * aCols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, aCols * bCols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, aRows * bCols * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, A, aRows * aCols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, aCols * bCols * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((bCols + 15) / 16, (aRows + 15) / 16);

    MatrixMultiplyKernel<<<gridSize, blockSize>>>(dA, dB, dC, aRows, aCols, bCols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, dC, aRows * bCols * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return true;
}

void MatrixMultiplyCpu(const float* A, const float* B, float* C,
                       int aRows, int aCols, int bCols) {
    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bCols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < aCols; k++) {
                sum += A[i * aCols + k] * B[k * bCols + j];
            }
            C[i * bCols + j] = sum;
        }
    }
}

void PrintMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}
