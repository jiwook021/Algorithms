/**
 * @file test_Histogram.cu
 * @brief Unit tests for the Histogram CUDA kernel.
 */

#include "Histogram.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_EQ(actual, expected, msg)                                       \
    do {                                                                       \
        if ((actual) != (expected)) {                                          \
            fprintf(stderr, "[FAIL] %s: expected %u, got %u\n",               \
                    msg, (unsigned)(expected), (unsigned)(actual));            \
            testsFailed++;                                                     \
        } else {                                                               \
            testsPassed++;                                                     \
        }                                                                      \
    } while (0)

void TestUniformData() {
    const int N = 256;
    unsigned char data[256];
    for (int i = 0; i < N; i++) data[i] = (unsigned char)i;

    unsigned int gpuHist[NUM_BINS], cpuHist[NUM_BINS];
    memset(gpuHist, 0, sizeof(gpuHist));

    unsigned char* dData;
    unsigned int* dHist;
    CUDA_CHECK(cudaMalloc(&dData, N));
    CUDA_CHECK(cudaMalloc(&dHist, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dData, data, N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dHist, 0, NUM_BINS * sizeof(unsigned int)));

    HistogramKernel<<<1, 256>>>(dData, dHist, N, NUM_BINS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gpuHist, dHist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    ComputeHistogramCpu(data, cpuHist, N, NUM_BINS);

    for (int i = 0; i < NUM_BINS; i++) {
        ASSERT_EQ(gpuHist[i], cpuHist[i], "UniformData");
        ASSERT_EQ(gpuHist[i], 1u, "UniformData: each bin == 1");
    }
    printf("[PASSED] TestUniformData\n");

    CUDA_CHECK(cudaFree(dData));
    CUDA_CHECK(cudaFree(dHist));
}

void TestAllSameBin() {
    const int N = 1000;
    unsigned char* data = (unsigned char*)malloc(N);
    for (int i = 0; i < N; i++) data[i] = 42;

    unsigned int gpuHist[NUM_BINS];
    memset(gpuHist, 0, sizeof(gpuHist));

    unsigned char* dData;
    unsigned int* dHist;
    CUDA_CHECK(cudaMalloc(&dData, N));
    CUDA_CHECK(cudaMalloc(&dHist, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dData, data, N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dHist, 0, NUM_BINS * sizeof(unsigned int)));

    HistogramKernel<<<4, 256>>>(dData, dHist, N, NUM_BINS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gpuHist, dHist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    ASSERT_EQ(gpuHist[42], (unsigned int)N, "AllSameBin: bin 42");
    for (int i = 0; i < NUM_BINS; i++) {
        if (i != 42) ASSERT_EQ(gpuHist[i], 0u, "AllSameBin: other bins");
    }
    printf("[PASSED] TestAllSameBin\n");

    free(data);
    CUDA_CHECK(cudaFree(dData));
    CUDA_CHECK(cudaFree(dHist));
}

void TestCpuGpuMatch() {
    const int N = 100000;
    unsigned char* data = (unsigned char*)malloc(N);
    srand(12345);
    for (int i = 0; i < N; i++) data[i] = rand() % 256;

    unsigned int gpuHist[NUM_BINS], cpuHist[NUM_BINS];
    memset(gpuHist, 0, sizeof(gpuHist));

    unsigned char* dData;
    unsigned int* dHist;
    CUDA_CHECK(cudaMalloc(&dData, N));
    CUDA_CHECK(cudaMalloc(&dHist, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dData, data, N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dHist, 0, NUM_BINS * sizeof(unsigned int)));

    int gridSize = (N + 255) / 256;
    HistogramKernel<<<gridSize, 256>>>(dData, dHist, N, NUM_BINS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gpuHist, dHist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    ComputeHistogramCpu(data, cpuHist, N, NUM_BINS);

    bool match = CompareHistograms(cpuHist, gpuHist, NUM_BINS);
    if (match) {
        testsPassed++;
        printf("[PASSED] TestCpuGpuMatch\n");
    } else {
        testsFailed++;
        printf("[FAIL] TestCpuGpuMatch\n");
    }

    free(data);
    CUDA_CHECK(cudaFree(dData));
    CUDA_CHECK(cudaFree(dHist));
}

int main() {
    TestUniformData();
    TestAllSameBin();
    TestCpuGpuMatch();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
