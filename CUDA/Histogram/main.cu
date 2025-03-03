/**
 * @file main.cu
 * @brief Driver for CUDA histogram computation with GPU vs CPU comparison.
 */

#include "Histogram.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define DEFAULT_DATA_SIZE 10000000

int main(int argc, char** argv) {
    int dataSize = (argc > 1) ? atoi(argv[1]) : DEFAULT_DATA_SIZE;
    if (dataSize <= 0) dataSize = DEFAULT_DATA_SIZE;

    printf("Computing histogram for %d data points with %d bins\n", dataSize, NUM_BINS);

    srand(time(NULL));

    // Allocate pinned host memory
    unsigned char* hData = NULL;
    CUDA_CHECK(cudaMallocHost(&hData, dataSize * sizeof(unsigned char)));
    for (int i = 0; i < dataSize; i++) {
        hData[i] = rand() % (MAX_PIXEL_VALUE + 1);
    }

    unsigned int* hHistGpu = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    unsigned int* hHistCpu = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    memset(hHistGpu, 0, NUM_BINS * sizeof(unsigned int));

    // Device memory
    unsigned char* dData = NULL;
    unsigned int* dHistogram = NULL;
    CUDA_CHECK(cudaMalloc(&dData, dataSize * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&dHistogram, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dData, hData, dataSize * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dHistogram, 0, NUM_BINS * sizeof(unsigned int)));

    // GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int gridSize = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (gridSize > 65535) gridSize = 65535;

    CUDA_CHECK(cudaEventRecord(start));
    HistogramKernel<<<gridSize, BLOCK_SIZE>>>(dData, dHistogram, dataSize, NUM_BINS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpuTime;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));
    CUDA_CHECK(cudaMemcpy(hHistGpu, dHistogram, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // CPU reference
    clock_t cpuStart = clock();
    ComputeHistogramCpu(hData, hHistCpu, dataSize, NUM_BINS);
    clock_t cpuEnd = clock();
    float cpuTime = 1000.0f * (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    bool match = CompareHistograms(hHistCpu, hHistGpu, NUM_BINS);

    printf("\nGPU time: %.3f ms\n", gpuTime);
    printf("CPU time: %.3f ms\n", cpuTime);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    printf("Verification: %s\n", match ? "PASSED" : "FAILED");

    // Cleanup
    CUDA_CHECK(cudaFree(dData));
    CUDA_CHECK(cudaFree(dHistogram));
    CUDA_CHECK(cudaFreeHost(hData));
    free(hHistGpu);
    free(hHistCpu);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
