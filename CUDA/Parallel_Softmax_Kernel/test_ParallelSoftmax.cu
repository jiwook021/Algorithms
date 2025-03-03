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
    // 1 pixel with 3 channels: {1.0, 2.0, 3.0}
    float hInput[] = {1.0f, 2.0f, 3.0f};
    float hOutput[3] = {};

    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hInput, 3 * sizeof(float), cudaMemcpyHostToDevice));

    SoftmaxKernel<<<1, 1>>>(dIn, dOut, 1, 1, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOutput, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = hOutput[0] + hOutput[1] + hOutput[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f, "SoftmaxSumsToOne");
    // Highest input should have highest probability
    if (hOutput[2] > hOutput[1] && hOutput[1] > hOutput[0]) testsPassed++;
    else { testsFailed++; fprintf(stderr, "[FAIL] Softmax ordering\n"); }

    printf("[PASSED] TestSoftmaxSumsToOne\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

void TestArgmax() {
    // 2 pixels, 3 channels: pixel0={1,2,3}, pixel1={5,3,1}
    float hSoftmax[] = {0.09f, 0.24f, 0.67f, 0.84f, 0.11f, 0.05f};
    int hArgmax[2] = {};

    float *dIn;
    int *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dIn, hSoftmax, 6 * sizeof(float), cudaMemcpyHostToDevice));

    ArgmaxKernel<<<1, 2>>>(dIn, dOut, 1, 2, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hArgmax, dOut, 2 * sizeof(int), cudaMemcpyDeviceToHost));

    if (hArgmax[0] == 2) testsPassed++;
    else { testsFailed++; fprintf(stderr, "[FAIL] Argmax[0]: expected 2, got %d\n", hArgmax[0]); }

    if (hArgmax[1] == 0) testsPassed++;
    else { testsFailed++; fprintf(stderr, "[FAIL] Argmax[1]: expected 0, got %d\n", hArgmax[1]); }

    printf("[PASSED] TestArgmax\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

void TestSoftmaxNumericalStability() {
    // Large values that could overflow without max subtraction
    float hInput[] = {1000.0f, 1001.0f, 1002.0f};
    float hOutput[3] = {};

    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hInput, 3 * sizeof(float), cudaMemcpyHostToDevice));

    SoftmaxKernel<<<1, 1>>>(dIn, dOut, 1, 1, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOutput, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = hOutput[0] + hOutput[1] + hOutput[2];
    ASSERT_NEAR(sum, 1.0f, 1e-4f, "NumericalStability: sum=1");

    // No NaN or Inf
    for (int i = 0; i < 3; i++) {
        if (std::isnan(hOutput[i]) || std::isinf(hOutput[i])) {
            testsFailed++;
            fprintf(stderr, "[FAIL] NaN/Inf at index %d\n", i);
        } else {
            testsPassed++;
        }
    }
    printf("[PASSED] TestSoftmaxNumericalStability\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

int main() {
    TestSoftmaxSumsToOne();
    TestArgmax();
    TestSoftmaxNumericalStability();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
