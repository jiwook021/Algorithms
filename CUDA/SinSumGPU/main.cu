/**
 * @file main.cu
 * @brief Driver for GPU-accelerated sine sum with trapezoidal integration.
 *
 * Usage: ./main [steps] [terms]
 */

#include "SinSumGpu.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

int main(int argc, char* argv[]) {
    int steps = (argc > 1) ? atoi(argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;

    int threads = 256;
    int blocks  = (steps + threads - 1) / threads;

    double pi       = 3.14159265358979323;
    double stepSize = pi / (steps - 1);

    float* dSums = NULL;
    float* dSum  = NULL;
    cudaMalloc(&dSums, steps * sizeof(float));
    cudaMalloc(&dSum, sizeof(float));
    cudaMemset(dSum, 0, sizeof(float));

    clock_t start = clock();

    GpuSinKernel<<<blocks, threads>>>(dSums, steps, terms, (float)stepSize);
    cudaDeviceSynchronize();

    SumArrayKernel<<<blocks, threads, threads * sizeof(float)>>>(dSums, steps, dSum);
    cudaDeviceSynchronize();

    float hSum;
    cudaMemcpy(&hSum, dSum, sizeof(float), cudaMemcpyDeviceToHost);

    // Trapezoidal rule correction
    double gpuSum = hSum;
    gpuSum -= 0.5 * (SinSum(0.0f, terms) + SinSum((float)pi, terms));
    gpuSum *= stepSize;

    clock_t end = clock();
    double gpuTime = 1000.0 * (double)(end - start) / CLOCKS_PER_SEC;

    printf("gpu sum = %.10f, steps %d terms %d time %.3f ms\n",
           gpuSum, steps, terms, gpuTime);

    cudaFree(dSums);
    cudaFree(dSum);

    return 0;
}
