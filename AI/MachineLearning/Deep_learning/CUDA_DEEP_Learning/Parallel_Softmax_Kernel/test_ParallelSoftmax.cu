/**
 * @file test_ParallelSoftmax.cu
 * @brief Unit tests for ParallelSoftmax CUDA kernels.
 */

#include "ParallelSoftmax.cuh"
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

void TestSoftmaxSumsToOne() {
    float hIn[] = {1.0f, 2.0f, 3.0f};
    float hOut[3] = {};
    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, 3 * sizeof(float), cudaMemcpyHostToDevice));

    SoftmaxKernel<<<1, 1>>>(dIn, dOut, 1, 1, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOut, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = hOut[0] + hOut[1] + hOut[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f, "Softmax sums to 1");
    printf("[PASSED] TestSoftmaxSumsToOne\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

void TestArgmax() {
    float hIn[] = {0.1f, 0.7f, 0.2f};
    int hOut[1] = {};
    float *dIn;
    int *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, 3 * sizeof(float), cudaMemcpyHostToDevice));

    ArgmaxKernel<<<1, 1>>>(dIn, dOut, 1, 1, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOut, dOut, sizeof(int), cudaMemcpyDeviceToHost));

    if (hOut[0] == 1) testsPassed++;
    else { testsFailed++; fprintf(stderr, "[FAIL] Argmax: expected 1, got %d\n", hOut[0]); }
    printf("[PASSED] TestArgmax\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

int main() {
    TestSoftmaxSumsToOne();
    TestArgmax();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
