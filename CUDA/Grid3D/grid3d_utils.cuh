/**
 * grid3d_utils.cuh — Shared utilities for the grid3D demonstration
 *
 * Contains:
 *   - CHECK_CUDA_ERROR macro   (wraps every CUDA API call)
 *   - linearize3D()            (3D coordinate -> flat index)
 *   - threadRankInBlock()      (local rank within a block)
 *   - blockRankInGrid()        (block's position in the grid)
 *   - threadRankInGrid()       (unique global thread rank)
 *
 * These are __host__ __device__ so they can be used both on GPU (in kernels)
 * and on CPU (in unit tests) to verify correctness.
 */

#ifndef GRID3D_UTILS_CUH
#define GRID3D_UTILS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#else
// Host-only stubs when compiling without nvcc
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * CHECK_CUDA_ERROR — Wraps a CUDA API call and aborts on failure
 * Only available when compiling with nvcc (CUDA toolkit)
 * ------------------------------------------------------------------------ */
#ifdef __CUDACC__
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
#endif

/* =========================================================================
 * linearize3D — Convert (x, y, z) to a flat 1D index
 *
 * Row-major order with z as the outermost dimension:
 *   memory: a[0][0][0], a[0][0][1], ..., a[nz-1][ny-1][nx-1]
 *   index = (z * ny + y) * nx + x
 * ======================================================================== */
__device__ __host__ inline int linearize3D(int x, int y, int z, int nx, int ny)
{
    return (z * ny + y) * nx + x;
}

/* =========================================================================
 * threadRankInBlock — Thread's linear position within its block
 *
 *   rank = (tz * blockDim.y + ty) * blockDim.x + tx
 *
 * For CPU testing, pass the dimensions explicitly.
 * ======================================================================== */
__host__ inline int threadRankInBlock(int tx, int ty, int tz,
                                      int bdx, int bdy)
{
    return (tz * bdy + ty) * bdx + tx;
}

/* =========================================================================
 * blockRankInGrid — Block's linear position within the grid
 *
 *   rank = (bz * gridDim.y + by) * gridDim.x + bx
 * ======================================================================== */
__host__ inline int blockRankInGrid(int bx, int by, int bz,
                                    int gdx, int gdy)
{
    return (bz * gdy + by) * gdx + bx;
}

/* =========================================================================
 * threadRankInGrid — Unique global rank across all threads
 *
 *   rank = threadRankInBlock + blockSize * blockRankInGrid
 * ======================================================================== */
__host__ inline int threadRankInGrid(int rankInBlock, int blockSize,
                                     int rankOfBlock)
{
    return rankInBlock + blockSize * rankOfBlock;
}

#endif // GRID3D_UTILS_CUH
