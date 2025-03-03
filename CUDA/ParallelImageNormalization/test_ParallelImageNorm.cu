/**
 * @file test_ParallelImageNorm.cu
 * @brief Unit tests for ParallelImageNorm CUDA kernels.
 */

#include "ParallelImageNorm.cuh"
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

void TestConvertUcharToFloat() {
    unsigned char hIn[] = {0, 128, 255};
    float hOut[3] = {};

    unsigned char* dIn;
    float* dOut;
    cudaMalloc(&dIn, 3);
    cudaMalloc(&dOut, 3 * sizeof(float));
    cudaMemcpy(dIn, hIn, 3, cudaMemcpyHostToDevice);

    ConvertUcharToFloat<<<1, 3>>>(dIn, dOut, 3);
    cudaDeviceSynchronize();
    cudaMemcpy(hOut, dOut, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    ASSERT_NEAR(hOut[0], 0.0f, 0.01f, "Convert[0]");
    ASSERT_NEAR(hOut[1], 128.0f, 0.01f, "Convert[128]");
    ASSERT_NEAR(hOut[2], 255.0f, 0.01f, "Convert[255]");
    printf("[PASSED] TestConvertUcharToFloat\n");

    cudaFree(dIn);
    cudaFree(dOut);
}

void TestNormalize() {
    // 4 pixels, 1 channel: values {10, 20, 30, 40} -> mean=25, std=~11.18
    float hIn[] = {10, 20, 30, 40};
    float mean[] = {25.0f};
    float stddev[] = {11.18f};
    float hOut[4] = {};

    float *dIn, *dMean, *dStd, *dOut;
    cudaMalloc(&dIn, 4 * sizeof(float));
    cudaMalloc(&dMean, sizeof(float));
    cudaMalloc(&dStd, sizeof(float));
    cudaMalloc(&dOut, 4 * sizeof(float));

    cudaMemcpy(dIn, hIn, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dMean, mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dStd, stddev, sizeof(float), cudaMemcpyHostToDevice);

    NormalizeKernel<<<1, 4>>>(dIn, dMean, dStd, dOut, 4, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(hOut, dOut, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: (10-25)/11.18 = -1.342, (20-25)/11.18 = -0.447, etc.
    ASSERT_NEAR(hOut[0], -1.342f, 0.02f, "Norm[0]");
    ASSERT_NEAR(hOut[1], -0.447f, 0.02f, "Norm[1]");
    ASSERT_NEAR(hOut[2],  0.447f, 0.02f, "Norm[2]");
    ASSERT_NEAR(hOut[3],  1.342f, 0.02f, "Norm[3]");
    printf("[PASSED] TestNormalize\n");

    cudaFree(dIn);
    cudaFree(dMean);
    cudaFree(dStd);
    cudaFree(dOut);
}

int main() {
    TestConvertUcharToFloat();
    TestNormalize();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
