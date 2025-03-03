/**
 * ===========================================================================
 * CUDA 3D Grid Demonstration
 * ===========================================================================
 *
 * Source: "Programming in Parallel with CUDA" by Richard Ansorge (Example 2.3)
 * License: CC BY-NC 4.0 for non-commercial use
 *
 * Purpose:
 *   This program demonstrates how CUDA organizes threads into a 3-dimensional
 *   grid of 3-dimensional thread blocks. It is an educational tool for
 *   understanding the CUDA execution model:
 *
 *   1. Thread Hierarchy
 *      CUDA launches a *grid* of *blocks*. Each block contains threads.
 *      Both grids and blocks can be 1D, 2D, or 3D.
 *
 *          Grid (of blocks)              Block (of threads)
 *        ┌──────────────────┐          ┌──────────────────┐
 *        │  gridDim.x = 16  │          │  blockDim.x = 32 │
 *        │  gridDim.y = 64  │          │  blockDim.y = 8  │
 *        │  gridDim.z = 128 │          │  blockDim.z = 2  │
 *        │  total = 131,072 │          │  total = 512     │
 *        └──────────────────┘          └──────────────────┘
 *
 *   2. Thread Identification
 *      Each thread computes its global (x, y, z) coordinate:
 *        x = blockIdx.x * blockDim.x + threadIdx.x
 *        y = blockIdx.y * blockDim.y + threadIdx.y
 *        z = blockIdx.z * blockDim.z + threadIdx.z
 *
 *   3. Linearization (3D -> 1D index)
 *      To map a 3D coordinate to a flat array index (row-major, z-outermost):
 *        idx = (z * ny + y) * nx + x
 *
 *      To compute a unique global "rank" for a thread:
 *        rank_in_block  = (tz * blockDim.y + ty) * blockDim.x + tx
 *        rank_of_block  = (bz * gridDim.y  + by) * gridDim.x  + bx
 *        rank_in_grid   = rank_in_block + block_size * rank_of_block
 *
 *   4. What the Kernel Does
 *      - Assigns each array element a[idx] = thread_rank_in_grid
 *      - Computes b[idx] = sqrt(a[idx])
 *      - For one selected thread (by id), prints all hierarchy info
 *
 * Usage:
 *   ./main [thread_id]       (default: 12345)
 *
 * Build:
 *   nvcc -O2 --expt-relaxed-constexpr main.cu -o main
 *
 * ===========================================================================
 */

#include "grid3d_utils.cuh"

/* =========================================================================
 * grid3D Kernel
 * =========================================================================
 * Each of the nx*ny*nz threads does:
 *   1. Computes its global 3D coordinate (x, y, z)
 *   2. Bounds-checks against the logical array dimensions
 *   3. Writes its unique rank into a[idx] and sqrt(rank) into b[idx]
 *   4. If this thread's rank matches `id`, prints diagnostics
 *
 * @param a   Device pointer — int array of size nx*ny*nz
 * @param b   Device pointer — float array of size nx*ny*nz
 * @param nx  X dimension (width,  mapped to threadIdx.x / blockIdx.x)
 * @param ny  Y dimension (height, mapped to threadIdx.y / blockIdx.y)
 * @param nz  Z dimension (depth,  mapped to threadIdx.z / blockIdx.z)
 * @param id  Which thread rank to print debug info for (-1 = none)
 * ======================================================================== */
__global__ void Grid3D(int* a, float* b, int Nx, int Ny, int Nz, int Id)
{
    /* --- Step 1: Global 3D coordinate for this thread --- */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    /* --- Step 2: Bounds check (grid may be larger than the array) --- */
    if (x >= Nx || y >= Ny || z >= Nz) return;

    /* --- Step 3: Compute hierarchy sizes for diagnostics --- */
    int ArraySize    = Nx * Ny * Nz;
    int BlockSize    = blockDim.x * blockDim.y * blockDim.z;
    int GridSize     = gridDim.x * gridDim.y * gridDim.z;
    int TotalThreads = BlockSize * GridSize;

    /* --- Step 4: Linearize thread position into unique ranks ---
     *  thread_rank_in_block: thread's position within its own block (0..block_size-1)
     *  block_rank_in_grid:   block's position within the grid       (0..grid_size-1)
     *  thread_rank_in_grid:  unique global rank across all threads  (0..total-1)
     */
    int ThreadRankInBlock =
        (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int BlockRankInGrid =
        (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    int ThreadRankInGrid =
        ThreadRankInBlock + BlockSize * BlockRankInGrid;

    /* --- Step 5: Write data — the actual "work" of this kernel --- */
    int Idx = (z * Ny + y) * Nx + x;
    a[Idx] = ThreadRankInGrid;
    b[Idx] = sqrtf((float)a[Idx]);

    /* --- Step 6: Print diagnostics for the selected thread --- */
    if (ThreadRankInGrid == Id) {
        printf("array size   %3d x %3d x %3d = %d\n", Nx, Ny, Nz, ArraySize);
        printf("thread block %3d x %3d x %3d = %d\n",
               blockDim.x, blockDim.y, blockDim.z, BlockSize);
        printf("thread  grid %3d x %3d x %3d = %d\n",
               gridDim.x, gridDim.y, gridDim.z, GridSize);
        printf("total number of threads in grid %d\n", TotalThreads);
        printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n",
               z, y, x, a[Idx], z, y, x, b[Idx]);
        printf("rank_in_block = %d rank_in_grid = %d block_rank_in_grid = %d\n",
               ThreadRankInBlock, ThreadRankInGrid, BlockRankInGrid);
    }
}

/* =========================================================================
 * Main — Allocates GPU memory, launches the kernel, cleans up
 * ======================================================================== */
int main(int argc, char* argv[])
{
    int Id = (argc > 1) ? atoi(argv[1]) : 12345;

    /* Configurable dimensions — change these or read from argv */
    int Nx = 512, Ny = 512, Nz = 256;
    size_t NumElements = (size_t)Nx * Ny * Nz;

    /* Dynamically allocate device memory (replaces static __device__ arrays) */
    int*   DA = nullptr;
    float* DB = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&DA, NumElements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&DB, NumElements * sizeof(float)));

    /*
     * Launch configuration:
     *   thread3d = (32, 8, 2)   -> 512 threads per block
     *   block3d  = (16, 64, 128) -> covers 512 x 512 x 256 elements
     *   Total threads = 512 * 131072 = 67,108,864 = nx*ny*nz
     */
    dim3 Thread3d(32, 8, 2);
    dim3 Block3d(16, 64, 128);

    Grid3D<<<Block3d, Thread3d>>>(DA, DB, Nx, Ny, Nz, Id);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaFree(DA));
    CHECK_CUDA_ERROR(cudaFree(DB));

    return 0;
}