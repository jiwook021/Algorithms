/**
 * @file test_ActivationFunction.cu
 * @brief Unit tests for ActivationFunction CUDA kernels.
 */

#include "ActivationFunction.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

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

void TestRelu() {
    float hIn[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float hOut[5] = {};
    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, 5 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, 5 * sizeof(float), cudaMemcpyHostToDevice));

    Relu(dIn, dOut, 5);
    CUDA_CHECK(cudaMemcpy(hOut, dOut, 5 * sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_NEAR(hOut[0], 0.0f, 1e-5f, "ReLU(-2)");
    ASSERT_NEAR(hOut[1], 0.0f, 1e-5f, "ReLU(-1)");
    ASSERT_NEAR(hOut[2], 0.0f, 1e-5f, "ReLU(0)");
    ASSERT_NEAR(hOut[3], 1.0f, 1e-5f, "ReLU(1)");
    ASSERT_NEAR(hOut[4], 2.0f, 1e-5f, "ReLU(2)");
    printf("[PASSED] TestRelu\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

void TestLeakyRelu() {
    float hIn[] = {-2.0f, 0.0f, 2.0f};
    float hOut[3] = {};
    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, 3 * sizeof(float), cudaMemcpyHostToDevice));

    LeakyRelu(dIn, dOut, 3, 0.01f);
    CUDA_CHECK(cudaMemcpy(hOut, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_NEAR(hOut[0], -0.02f, 1e-5f, "LeakyReLU(-2)");
    ASSERT_NEAR(hOut[1], 0.0f, 1e-5f, "LeakyReLU(0)");
    ASSERT_NEAR(hOut[2], 2.0f, 1e-5f, "LeakyReLU(2)");
    printf("[PASSED] TestLeakyRelu\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

void TestSoftmaxSumsToOne() {
    float hIn[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hOut[4] = {};
    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, 4 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, 4 * sizeof(float), cudaMemcpyHostToDevice));

    Softmax(dIn, dOut, 1, 4);
    CUDA_CHECK(cudaMemcpy(hOut, dOut, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = hOut[0] + hOut[1] + hOut[2] + hOut[3];
    ASSERT_NEAR(sum, 1.0f, 1e-4f, "Softmax sum=1");
    printf("[PASSED] TestSoftmaxSumsToOne\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

void TestGelu() {
    float hIn[] = {0.0f};
    float hOut[1] = {};
    float *dIn, *dOut;
    CUDA_CHECK(cudaMalloc(&dIn, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIn, hIn, sizeof(float), cudaMemcpyHostToDevice));

    Gelu(dIn, dOut, 1);
    CUDA_CHECK(cudaMemcpy(hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_NEAR(hOut[0], 0.0f, 1e-5f, "GELU(0)");
    printf("[PASSED] TestGelu\n");

    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dOut));
}

int main() {
    TestRelu();
    TestLeakyRelu();
    TestSoftmaxSumsToOne();
    TestGelu();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
