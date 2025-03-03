#ifndef DEAD_LOCK_CUH
#define DEAD_LOCK_CUH

/**
 * @file DeadLock.cuh
 * @brief Header for CUDA deadlock demonstration kernel.
 *
 * Demonstrates how conditional __syncthreads() and spin-locks cause deadlock
 * when threads in the same warp diverge under the SIMT execution model.
 *
 * Reference: "Programming in Parallel with CUDA" by Richard Ansorge (Example 3.8).
 */

#include <cuda_runtime.h>

/**
 * @brief Kernel that demonstrates warp-level deadlock via divergent synchronization.
 *
 * Threads are split into three groups (A, B, C):
 *   - Group A (threadIdx.x < gsync):            enters one __syncthreads() branch.
 *   - Group B (gsync <= threadIdx.x < 2*gsync): enters another __syncthreads() branch.
 *   - Group C (all):                            optionally spin-waits on a shared lock.
 *
 * If Group A and Group C are in the same warp, Group C spin-waits on `lock`
 * while Group A is stuck at __syncthreads() -- deadlock.
 *
 * @param gsync   Boundary for group division (threads 0..gsync-1 = Group A).
 * @param doLock  If nonzero, enable the spin-lock (Group C waits on lock).
 */
__global__ void DeadLockKernel(int gsync, int doLock);

#endif // DEAD_LOCK_CUH
