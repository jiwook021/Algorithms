/**
 * @file main.cu
 * @brief Driver for CUDA parallel reduction with CPU comparison.
 *
 * Usage: ./main [N]  (N defaults to 2^24 = 16,777,216, must be power of 2)
 */

#include "ParallelReduction.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>

double GetTimeMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

float RandFloat() {
    return (float)rand() / (float)RAND_MAX;
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 24;

    float* hX = (float*)malloc(N * sizeof(float));
    float* dX = NULL;
    CHECK_CUDA(cudaMalloc(&dX, N * sizeof(float)));

    srand(12345678);
    for (int k = 0; k < N; k++) {
        hX[k] = RandFloat();
    }

    CHECK_CUDA(cudaMemcpy(dX, hX, N * sizeof(float), cudaMemcpyHostToDevice));

    // CPU reduction
    double tStart = GetTimeMs();
    double hostSum = 0.0;
    for (int k = 0; k < N; k++) {
        hostSum += hX[k];
    }
    double t1 = GetTimeMs() - tStart;

    // GPU reduction
    tStart = GetTimeMs();
    for (int m = N / 2; m > 0; m /= 2) {
        int threads = (m < 256) ? m : 256;
        int blocks  = (m / 256) > 0 ? (m / 256) : 1;
        ReduceKernel<<<blocks, threads>>>(dX, m);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double t2 = GetTimeMs() - tStart;

    float gpuSum = 0.0f;
    CHECK_CUDA(cudaMemcpy(&gpuSum, dX, sizeof(float), cudaMemcpyDeviceToHost));

    printf("sum of %d random numbers: host %.1f %.3f ms, GPU %.1f %.3f ms\n",
           N, hostSum, t1, (double)gpuSum, t2);

    free(hX);
    CHECK_CUDA(cudaFree(dX));

    return 0;
}
