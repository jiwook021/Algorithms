/**
 * @file test_ParallelReduction.cu
 * @brief Unit tests for the ParallelReduction CUDA kernel.
 */

#include "ParallelReduction.cuh"
#include <cstdio>
#include <cmath>

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

void TestPowerOf2Sum() {
    const int N = 1024;
    float hX[1024];
    float expectedSum = 0.0f;
    for (int i = 0; i < N; i++) {
        hX[i] = 1.0f;
        expectedSum += 1.0f;
    }

    float* dX;
    CHECK_CUDA(cudaMalloc(&dX, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dX, hX, N * sizeof(float), cudaMemcpyHostToDevice));

    for (int m = N / 2; m > 0; m /= 2) {
        int threads = (m < 256) ? m : 256;
        int blocks  = (m + threads - 1) / threads;
        ReduceKernel<<<blocks, threads>>>(dX, m);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    float result;
    CHECK_CUDA(cudaMemcpy(&result, dX, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_NEAR(result, expectedSum, 1.0f, "PowerOf2Sum: 1024 ones");
    printf("[PASSED] TestPowerOf2Sum: result=%.1f\n", result);

    CHECK_CUDA(cudaFree(dX));
}

void TestSmallArray() {
    const int N = 8;
    float hX[] = {1, 2, 3, 4, 5, 6, 7, 8};  // sum = 36

    float* dX;
    CHECK_CUDA(cudaMalloc(&dX, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dX, hX, N * sizeof(float), cudaMemcpyHostToDevice));

    for (int m = N / 2; m > 0; m /= 2) {
        ReduceKernel<<<1, m>>>(dX, m);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    float result;
    CHECK_CUDA(cudaMemcpy(&result, dX, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_NEAR(result, 36.0f, 0.01f, "SmallArray: sum of 1..8");
    printf("[PASSED] TestSmallArray: result=%.1f\n", result);

    CHECK_CUDA(cudaFree(dX));
}

int main() {
    TestSmallArray();
    TestPowerOf2Sum();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
