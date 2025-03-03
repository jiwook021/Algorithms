/**
 * @file test_MatrixMultiply.cu
 * @brief Unit tests for the MatrixMultiply CUDA kernels.
 */

#include "MatrixMultiply.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

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

void TestSmall2x3x2() {
    float A[] = {1,2,3, 4,5,6};
    float B[] = {7,8, 9,10, 11,12};
    float cGpu[4] = {}, cCpu[4] = {};

    MatrixMultiplyCuda(A, B, cGpu, 2, 3, 2);
    MatrixMultiplyCpu(A, B, cCpu, 2, 3, 2);

    // Expected: [58 64; 139 154]
    ASSERT_NEAR(cGpu[0], 58.0f, 1e-3f, "2x3*3x2 [0,0]");
    ASSERT_NEAR(cGpu[1], 64.0f, 1e-3f, "2x3*3x2 [0,1]");
    ASSERT_NEAR(cGpu[2], 139.0f, 1e-3f, "2x3*3x2 [1,0]");
    ASSERT_NEAR(cGpu[3], 154.0f, 1e-3f, "2x3*3x2 [1,1]");

    for (int i = 0; i < 4; i++)
        ASSERT_NEAR(cGpu[i], cCpu[i], 1e-3f, "CPU/GPU match");

    printf("[PASSED] TestSmall2x3x2\n");
}

void TestIdentity() {
    float A[] = {1,0, 0,1};
    float B[] = {5,6, 7,8};
    float C[4] = {};

    MatrixMultiplyCuda(A, B, C, 2, 2, 2);

    ASSERT_NEAR(C[0], 5.0f, 1e-3f, "Identity [0,0]");
    ASSERT_NEAR(C[1], 6.0f, 1e-3f, "Identity [0,1]");
    ASSERT_NEAR(C[2], 7.0f, 1e-3f, "Identity [1,0]");
    ASSERT_NEAR(C[3], 8.0f, 1e-3f, "Identity [1,1]");

    printf("[PASSED] TestIdentity\n");
}

void TestTiledVsNaive() {
    const int N = 64;
    std::vector<float> A(N*N), B(N*N), cNaive(N*N, 0), cTiled(N*N, 0);

    srand(42);
    for (int i = 0; i < N*N; i++) {
        A[i] = (float)(rand() % 10);
        B[i] = (float)(rand() % 10);
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((N+15)/16, (N+15)/16);

    MatrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, N, N, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(cNaive.data(), dC, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    MatrixMultiplyTiledKernel<<<grid, block>>>(dA, dB, dC, N, N, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(cTiled.data(), dC, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N*N; i++) {
        ASSERT_NEAR(cNaive[i], cTiled[i], 0.01f, "TiledVsNaive");
    }
    printf("[PASSED] TestTiledVsNaive\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

int main() {
    TestSmall2x3x2();
    TestIdentity();
    TestTiledVsNaive();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
