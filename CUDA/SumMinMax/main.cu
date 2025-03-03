/**
 * @file main.cu
 * @brief Driver for CUDA Sum/Min/Max parallel reduction.
 */

#include "SumMinMax.cuh"
#include <cstdio>
#include <cstdlib>
#include <cfloat>

int main() {
    const int N = 1 << 20;  // 1M elements
    const int BLOCK_SIZE = 256;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t bytes = N * sizeof(float);

    float* hIn = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) {
        hIn[i] = (float)(rand() % 100) / 100.0f;
    }

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, gridSize * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    // Sum
    ReduceSumKernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(dIn, dOut, N);
    CHECK_CUDA(cudaGetLastError());
    float* hSum = (float*)malloc(gridSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(hSum, dOut, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    float finalSum = 0.0f;
    for (int i = 0; i < gridSize; i++) finalSum += hSum[i];
    printf("Sum: %f\n", finalSum);

    // Min
    ReduceMinKernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(dIn, dOut, N);
    CHECK_CUDA(cudaGetLastError());
    float* hMin = (float*)malloc(gridSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(hMin, dOut, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    float finalMin = FLT_MAX;
    for (int i = 0; i < gridSize; i++) if (hMin[i] < finalMin) finalMin = hMin[i];
    printf("Min: %f\n", finalMin);

    // Max
    ReduceMaxKernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(dIn, dOut, N);
    CHECK_CUDA(cudaGetLastError());
    float* hMax = (float*)malloc(gridSize * sizeof(float));
    CHECK_CUDA(cudaMemcpy(hMax, dOut, gridSize * sizeof(float), cudaMemcpyDeviceToHost));
    float finalMax = -FLT_MAX;
    for (int i = 0; i < gridSize; i++) if (hMax[i] > finalMax) finalMax = hMax[i];
    printf("Max: %f\n", finalMax);

    free(hIn); free(hSum); free(hMin); free(hMax);
    cudaFree(dIn); cudaFree(dOut);
    return 0;
}
