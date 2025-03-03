/**
 * @file main.cu
 * @brief Driver for CUDA matrix multiplication with naive vs tiled benchmark.
 */

#include "MatrixMultiply.cuh"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main() {
    // Small correctness test
    const int A_ROWS = 2, A_COLS = 3, B_COLS = 2;
    float aSmall[] = {1,2,3, 4,5,6};
    float bSmall[] = {7,8, 9,10, 11,12};
    float cSmall[4] = {}, cCpuSmall[4] = {};

    std::cout << "=== Small Matrix Test (2x3 * 3x2) ===" << std::endl;
    MatrixMultiplyCuda(aSmall, bSmall, cSmall, A_ROWS, A_COLS, B_COLS);
    MatrixMultiplyCpu(aSmall, bSmall, cCpuSmall, A_ROWS, A_COLS, B_COLS);
    PrintMatrix(cSmall, 2, 2);

    bool smallMatch = true;
    for (int i = 0; i < 4; i++)
        if (std::abs(cSmall[i] - cCpuSmall[i]) > 1e-5f) smallMatch = false;
    std::cout << "Small test: " << (smallMatch ? "PASSED" : "FAILED") << std::endl;

    // Large benchmark
    const int N = 512;
    std::vector<float> aBig(N*N), bBig(N*N), cNaive(N*N), cTiled(N*N), cRef(N*N);

    srand(42);
    for (int i = 0; i < N*N; i++) {
        aBig[i] = (float)(rand() % 100) / 100.0f;
        bBig[i] = (float)(rand() % 100) / 100.0f;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, aBig.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, bBig.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((N+15)/16, (N+15)/16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    MatrixMultiplyKernel<<<gridSize, blockSize>>>(dA, dB, dC, N, N, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float naiveMs;
    CUDA_CHECK(cudaEventElapsedTime(&naiveMs, start, stop));
    CUDA_CHECK(cudaMemcpy(cNaive.data(), dC, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(start));
    MatrixMultiplyTiledKernel<<<gridSize, blockSize>>>(dA, dB, dC, N, N, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float tiledMs;
    CUDA_CHECK(cudaEventElapsedTime(&tiledMs, start, stop));
    CUDA_CHECK(cudaMemcpy(cTiled.data(), dC, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    auto cpuStart = std::chrono::high_resolution_clock::now();
    MatrixMultiplyCpu(aBig.data(), bBig.data(), cRef.data(), N, N, N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuMs = cpuEnd - cpuStart;

    bool naiveOk = true, tiledOk = true;
    for (int i = 0; i < N*N; i++) {
        if (std::abs(cNaive[i] - cRef[i]) > 0.01f) naiveOk = false;
        if (std::abs(cTiled[i] - cRef[i]) > 0.01f) tiledOk = false;
    }

    std::cout << "\n=== " << N << "x" << N << " Benchmark ===" << std::endl;
    std::cout << "CPU:          " << cpuMs.count() << " ms" << std::endl;
    std::cout << "GPU (naive):  " << naiveMs << " ms  [" << (naiveOk ? "PASS" : "FAIL") << "]" << std::endl;
    std::cout << "GPU (tiled):  " << tiledMs << " ms  [" << (tiledOk ? "PASS" : "FAIL") << "]" << std::endl;

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
