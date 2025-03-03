/**
 * @file test_ThreadIdx.cu
 * @brief Unit tests for the ThreadIdx demonstration kernel.
 *
 * Verifies that the kernel launches successfully with various configurations.
 */

#include "ThreadIdx.cuh"
#include <cstdio>

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_CUDA_OK(msg)                                                    \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "[FAIL] %s: %s\n", msg, cudaGetErrorString(err));  \
            testsFailed++;                                                     \
        } else {                                                               \
            testsPassed++;                                                     \
            printf("[PASSED] %s\n", msg);                                      \
        }                                                                      \
    } while (0)

void TestSingleBlock() {
    PrintThreadIdx<<<1, 32>>>();
    cudaDeviceSynchronize();
    ASSERT_CUDA_OK("SingleBlock: 1 block, 32 threads");
}

void TestMultipleBlocks() {
    PrintThreadIdx<<<4, 64>>>();
    cudaDeviceSynchronize();
    ASSERT_CUDA_OK("MultipleBlocks: 4 blocks, 64 threads");
}

void TestSmallConfig() {
    PrintThreadIdx<<<2, 30>>>();
    cudaDeviceSynchronize();
    ASSERT_CUDA_OK("SmallConfig: 2 blocks, 30 threads");
}

int main() {
    TestSingleBlock();
    TestMultipleBlocks();
    TestSmallConfig();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
