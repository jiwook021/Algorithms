/**
 * @file test_SumMinMax.cu
 * @brief Unit tests for the SumMinMax CUDA reduction kernels.
 */

#include "SumMinMax.cuh"
#include <cstdio>
#include <cmath>
#include <cfloat>

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_NEAR(actual, expected, tol, msg)                                \
    do {                                                                       \
        if (std::abs((actual) - (expected)) > (tol)) {                        \
            fprintf(stderr, "[FAIL] %s: expected %.4f, got %.4f\n",           \
                    msg, (float)(expected), (float)(actual));                  \
            testsFailed++;                                                     \
        } else {                                                               \
            testsPassed++;                                                     \
        }                                                                      \
    } while (0)

void TestSumKnown() {
    const int N = 256;
    float hIn[256];
    float expected = 0.0f;
    for (int i = 0; i < N; i++) { hIn[i] = (float)(i + 1); expected += hIn[i]; }

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice));

    ReduceSumKernel<<<1, 256, 256 * sizeof(float)>>>(dIn, dOut, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float result;
    CHECK_CUDA(cudaMemcpy(&result, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_NEAR(result, expected, 1.0f, "SumKnown: sum(1..256)");
    printf("[PASSED] TestSumKnown: result=%.1f\n", result);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
}

void TestMinMax() {
    const int N = 256;
    float hIn[256];
    for (int i = 0; i < N; i++) hIn[i] = (float)(i * 3 - 100);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice));

    // Min
    ReduceMinKernel<<<1, 256, 256 * sizeof(float)>>>(dIn, dOut, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float minResult;
    CHECK_CUDA(cudaMemcpy(&minResult, dOut, sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_NEAR(minResult, -100.0f, 0.01f, "Min: should be -100");

    // Max
    ReduceMaxKernel<<<1, 256, 256 * sizeof(float)>>>(dIn, dOut, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    float maxResult;
    CHECK_CUDA(cudaMemcpy(&maxResult, dOut, sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_NEAR(maxResult, 255.0f * 3.0f - 100.0f, 0.01f, "Max");

    printf("[PASSED] TestMinMax\n");

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
}

int main() {
    TestSumKnown();
    TestMinMax();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
