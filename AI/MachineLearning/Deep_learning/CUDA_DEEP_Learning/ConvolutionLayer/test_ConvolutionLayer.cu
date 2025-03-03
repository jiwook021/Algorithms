/**
 * @file test_ConvolutionLayer.cu
 * @brief Unit tests for the ConvolutionLayer CUDA kernel.
 */

#include "ConvolutionLayer.cuh"
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

void TestIdentityKernel() {
    // 3x3 input convolved with identity kernel [[0,0,0],[0,1,0],[0,0,0]] -> 1x1 output = center
    float hInput[] = {1,2,3, 4,5,6, 7,8,9};
    float hKernel[] = {0,0,0, 0,1,0, 0,0,0};
    float hOutput[1] = {};

    float *dIn, *dK, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hInput, 9 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hKernel, 9 * sizeof(float), cudaMemcpyHostToDevice));

    Conv2DKernel<<<1, 1>>>(dIn, dK, dOut, 3, 3, 3, 3);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOutput, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_NEAR(hOutput[0], 5.0f, 1e-5f, "IdentityKernel center=5");
    printf("[PASSED] TestIdentityKernel\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dOut));
}

void TestSumKernel() {
    // 4x4 input with all 1s, 2x2 kernel of all 1s -> 3x3 output of all 4s
    float hInput[16], hKernel[4];
    for (int i = 0; i < 16; i++) hInput[i] = 1.0f;
    for (int i = 0; i < 4; i++) hKernel[i] = 1.0f;
    float hOutput[9] = {};

    float *dIn, *dK, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, 9 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hInput, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hKernel, 4 * sizeof(float), cudaMemcpyHostToDevice));

    Conv2DKernel<<<1, dim3(3, 3)>>>(dIn, dK, dOut, 4, 4, 2, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hOutput, dOut, 9 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 9; i++) {
        ASSERT_NEAR(hOutput[i], 4.0f, 1e-5f, "SumKernel all 4s");
    }
    printf("[PASSED] TestSumKernel\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dOut));
}

int main() {
    TestIdentityKernel();
    TestSumKernel();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
