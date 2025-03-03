/**
 * @file test_DeadLock.cu
 * @brief Unit tests for the DeadLock demonstration kernel.
 *
 * Since the deadlock kernel is a demonstration of synchronization hazards,
 * these tests verify the safe (non-deadlocking) configurations complete
 * successfully.
 */

#include "DeadLock.cuh"
#include <cstdio>
#include <cstdlib>

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

/**
 * @brief Test: Kernel completes without deadlock when doLock=0 (lock disabled).
 */
void TestNoLock() {
    DeadLockKernel<<<1, 96>>>(32, 0);
    cudaDeviceSynchronize();
    ASSERT_CUDA_OK("NoLock: kernel completes with doLock=0");
}

/**
 * @brief Test: Kernel completes when gsync equals full block (no divergence).
 */
void TestFullBlockSync() {
    DeadLockKernel<<<1, 64>>>(64, 0);
    cudaDeviceSynchronize();
    ASSERT_CUDA_OK("FullBlockSync: gsync covers entire block");
}

/**
 * @brief Test: Multiple blocks with lock disabled.
 */
void TestMultipleBlocksNoLock() {
    DeadLockKernel<<<4, 128>>>(32, 0);
    cudaDeviceSynchronize();
    ASSERT_CUDA_OK("MultipleBlocksNoLock: 4 blocks, doLock=0");
}

/**
 * @brief Test: Single warp, lock disabled -- smallest safe configuration.
 */
void TestSingleWarpNoLock() {
    DeadLockKernel<<<1, 32>>>(16, 0);
    cudaDeviceSynchronize();
    ASSERT_CUDA_OK("SingleWarpNoLock: 1 warp, gsync=16, doLock=0");
}

int main() {
    TestNoLock();
    TestFullBlockSync();
    TestMultipleBlocksNoLock();
    TestSingleWarpNoLock();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
