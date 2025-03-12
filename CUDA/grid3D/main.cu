// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement
// grid3D example 2.3 - 3D grid computation demonstration
// This program creates a 3D grid of threads and performs a simple computation
// on two 3D arrays, assigning each element a value based on the thread's position

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Device arrays - allocated in GPU global memory
// Space complexity: O(nx*ny*nz) = O(2^27) ≈ 134 million elements
__device__ int   a[256][512][512];  // Integer array for storing thread ranks
__device__ float b[256][512][512];  // Float array for storing square root values

/**
 * @brief CUDA kernel to populate 3D arrays and print thread information
 * 
 * @param nx X dimension size (width)
 * @param ny Y dimension size (height)
 * @param nz Z dimension size (depth)
 * @param id Thread ID to print detailed information for
 * 
 * Time complexity: O(1) per thread, O(nx*ny*nz) total in parallel
 */
__global__ void grid3D(int nx, int ny, int nz, int id)
{
    // Calculate the global coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Early return if coordinates are out of bounds
    if (x >= nx || y >= ny || z >= nz) return;

    // Calculate various metrics for thread and block organization
    int array_size = nx * ny * nz;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    int grid_size = gridDim.x * gridDim.y * gridDim.z;
    int total_threads = block_size * grid_size;
    
    // Calculate linear indices for the thread within its block and the grid
    int thread_rank_in_block = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int block_rank_in_grid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    int thread_rank_in_grid = thread_rank_in_block + block_size * block_rank_in_grid;

    // Perform the computational work
    a[z][y][x] = thread_rank_in_grid;
    b[z][y][x] = sqrtf((float)a[z][y][x]);
    
    // Print information for the thread with ID matching the input parameter
    if (thread_rank_in_grid == id) {
        printf("array size   %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);
        printf("thread block %3d x %3d x %3d = %d\n", blockDim.x, blockDim.y, blockDim.z, block_size);
        printf("thread  grid %3d x %3d x %3d = %d\n", gridDim.x, gridDim.y, gridDim.z, grid_size);
        printf("total number of threads in grid %d\n", total_threads);
        printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n", z, y, x, a[z][y][x], z, y, x, b[z][y][x]);
        printf("rank_in_block = %d rank_in_grid = %d rank of block_rank_in_grid = %d\n", thread_rank_in_block, thread_rank_in_grid, block_rank_in_grid);
    }
}

/**
 * @brief Main function to set up and launch the CUDA kernel
 * 
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return int 0 on success, non-zero on failure
 */
int main(int argc, char* argv[])
{
    // Parse command-line arguments - default ID is 12345 if none provided
    int id = (argc > 1) ? atoi(argv[1]) : 12345;
    
    // Define the 3D thread and block dimensions
    dim3 thread3d(32, 8, 2);    // 32*8*2    = 512 threads per block
    dim3 block3d(16, 64, 128);  // 16*64*128 = 131072 blocks in grid
    
    // Launch the kernel
    grid3D<<<block3d, thread3d>>>(512, 512, 256, id);
    
    // Wait for kernel completion to see printf output
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Check for any asynchronous errors during kernel execution
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    return 0;
}