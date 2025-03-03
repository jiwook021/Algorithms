/**
 * @file test_SinSumGpu.cu
 * @brief Unit tests for SinSumGpu Taylor series and reduction kernels.
 */

#include "SinSumGpu.cuh"
#include <cstdio>
#include <cmath>

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_NEAR(actual, expected, tol, msg)                                \
    do {                                                                       \
        if (std::abs((actual) - (expected)) > (tol)) {                        \
            fprintf(stderr, "[FAIL] %s: expected %.6f, got %.6f\n",           \
                    msg, (float)(expected), (float)(actual));                  \
            testsFailed++;                                                     \
        } else {                                                               \
            testsPassed++;                                                     \
        }                                                                      \
    } while (0)

void TestSinSumCpu() {
    // sin(pi/2) should be 1.0
    float result = SinSum(3.14159265f / 2.0f, 100);
    ASSERT_NEAR(result, 1.0f, 1e-5f, "SinSum(pi/2, 100)");

    // sin(0) should be 0.0
    result = SinSum(0.0f, 100);
    ASSERT_NEAR(result, 0.0f, 1e-5f, "SinSum(0, 100)");

    // sin(pi) should be ~0.0
    result = SinSum(3.14159265f, 100);
    ASSERT_NEAR(result, 0.0f, 1e-5f, "SinSum(pi, 100)");

    printf("[PASSED] TestSinSumCpu\n");
}

void TestGpuKernel() {
    const int N = 256;
    float stepSize = 3.14159265f / (N - 1);

    float* dSums;
    cudaMalloc(&dSums, N * sizeof(float));
    GpuSinKernel<<<1, N>>>(dSums, N, 100, stepSize);
    cudaDeviceSynchronize();

    float hSums[256];
    cudaMemcpy(hSums, dSums, N * sizeof(float), cudaMemcpyDeviceToHost);

    // sin(0) ~ 0
    ASSERT_NEAR(hSums[0], 0.0f, 1e-5f, "GpuKernel: sin(0)");
    // sin(pi/2) ~ 1
    int midIdx = (N - 1) / 2;
    ASSERT_NEAR(hSums[midIdx], 1.0f, 0.01f, "GpuKernel: sin(pi/2)");

    printf("[PASSED] TestGpuKernel\n");

    cudaFree(dSums);
}

int main() {
    TestSinSumCpu();
    TestGpuKernel();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
