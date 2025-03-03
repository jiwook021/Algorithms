/**
 * @file test_BatchNormalization.cu
 * @brief Unit tests for BatchNormalization CUDA kernels.
 */

#include "BatchNormalization.cuh"
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

void TestMeanComputation() {
    // 2 batches, 1 feature, 2 spatial: values {1,2,3,4} -> mean=2.5
    float hInput[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hMean[1] = {};

    float *dInput, *dMean;
    CUDA_CHECK(cudaMalloc(&dInput, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMean, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dInput, hInput, 4 * sizeof(float), cudaMemcpyHostToDevice));

    ComputeMeanKernel<<<1, 1>>>(dInput, dMean, 2, 1, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hMean, dMean, sizeof(float), cudaMemcpyDeviceToHost));

    ASSERT_NEAR(hMean[0], 2.5f, 1e-5f, "Mean of {1,2,3,4}");
    printf("[PASSED] TestMeanComputation\n");

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dMean));
}

void TestForwardNormalization() {
    // 1 batch, 1 feature, 4 spatial: {-2,-1,1,2} with gamma=1, beta=0
    float hInput[] = {-2.0f, -1.0f, 1.0f, 2.0f};
    float gamma[] = {1.0f};
    float beta[] = {0.0f};

    float *dInput, *dOutput, *dMean, *dVar, *dGamma, *dBeta;
    CUDA_CHECK(cudaMalloc(&dInput, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutput, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMean, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dVar, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGamma, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBeta, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dInput, hInput, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dGamma, gamma, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBeta, beta, sizeof(float), cudaMemcpyHostToDevice));

    ComputeMeanKernel<<<1, 1>>>(dInput, dMean, 1, 1, 4);
    ComputeVarianceKernel<<<1, 1>>>(dInput, dMean, dVar, 1, 1, 4);
    BatchNormForwardKernel<<<1, 4>>>(dInput, dOutput, dMean, dVar, dGamma, dBeta,
                                     BN_EPSILON, 1, 1, 4);
    CUDA_CHECK(cudaDeviceSynchronize());

    float hOutput[4];
    CUDA_CHECK(cudaMemcpy(hOutput, dOutput, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // Mean=0, Var=2.5 -> normalized values should be symmetric
    ASSERT_NEAR(hOutput[0], -hOutput[3], 0.01f, "Symmetric normalization");
    ASSERT_NEAR(hOutput[1], -hOutput[2], 0.01f, "Symmetric normalization");
    printf("[PASSED] TestForwardNormalization\n");

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));
    CUDA_CHECK(cudaFree(dMean));
    CUDA_CHECK(cudaFree(dVar));
    CUDA_CHECK(cudaFree(dGamma));
    CUDA_CHECK(cudaFree(dBeta));
}

int main() {
    TestMeanComputation();
    TestForwardNormalization();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
