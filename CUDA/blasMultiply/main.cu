// Matrix multiplication implementation using standard CUDA kernels
// Based on the original cuBLAS benchmark code

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define block size for tiled matrix multiplication
#define BLOCK_SIZE 32

// CUDA kernel for basic matrix multiplication (C = A * B)
__global__ void matrixMultiplyBasic(float *A, float *B, float *C, 
                                   int Arows, int Acols, int Bcols) {
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if we're within the matrix dimensions
    if (row < Arows && col < Bcols) {
        float sum = 0.0f;
        // Multiply row of A by column of B and accumulate
        for (int k = 0; k < Acols; k++) {
            sum += A[row * Acols + k] * B[k * Bcols + col];
        }
        C[row * Bcols + col] = sum;
    }
}

// Optimized CUDA kernel using shared memory for tiled matrix multiplication
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, 
                                   int Arows, int Acols, int Bcols) {
    // Shared memory for tiles of A and B
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Thread indices within the block
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles of A and B
    for (int t = 0; t < (Acols + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load tiles into shared memory
        if (row < Arows && t * BLOCK_SIZE + tx < Acols) {
            sharedA[ty][tx] = A[row * Acols + t * BLOCK_SIZE + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (t * BLOCK_SIZE + ty < Acols && col < Bcols) {
            sharedB[ty][tx] = B[(t * BLOCK_SIZE + ty) * Bcols + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < Arows && col < Bcols) {
        C[row * Bcols + col] = sum;
    }
}

// Simple CUDA timer using CUDA events
class CudaTimer {
private:
    cudaEvent_t start, stop;
    
public:
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    double lap_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return static_cast<double>(milliseconds);
    }
};

// CUDA error checking helper
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main(int argc, char* argv[])
{
    // Parse command line arguments
    int Arow = (argc > 1) ? atoi(argv[1]) : 1024; // Default 1024
    int Acol = (argc > 2) ? atoi(argv[2]) : Arow;
    int Brow = Acol;
    int Bcol = (argc > 3) ? atoi(argv[3]) : Brow;
    int Crow = Arow;
    int Ccol = Bcol;
    int useOptimized = (argc > 4) ? atoi(argv[4]) : 1; // Use optimized kernel by default
    int nacc = (argc > 5) ? atoi(argv[5]) : 10;        // Number of runs for timing (reduced from 100)
    
    printf("Matrix sizes: A(%d×%d) × B(%d×%d) = C(%d×%d)\n", 
           Arow, Acol, Brow, Bcol, Crow, Ccol);
    
    // Allocate host memory
    float *A = new float[Arow * Acol];
    float *B = new float[Brow * Bcol];
    float *C = new float[Crow * Ccol];
    
    // Allocate device memory
    float *dev_A, *dev_B, *dev_C;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_A, Arow * Acol * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_B, Brow * Bcol * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_C, Crow * Ccol * sizeof(float)));

    // Initialize matrices with random values
    srand(12345678);
    for (int i = 0; i < Arow * Acol; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < Brow * Bcol; i++) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(dev_A, A, Arow * Acol * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_B, B, Brow * Bcol * sizeof(float), cudaMemcpyHostToDevice));
    
    // Set up grid and block dimensions for kernel launch
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((Ccol + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (Crow + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Start timing
    CudaTimer timer;
    
    // Perform matrix multiplication multiple times for timing
    for (int k = 0; k < nacc; k++) {
        if (useOptimized) {
            // Run the optimized tiled version
            matrixMultiplyTiled<<<gridDim, blockDim>>>(dev_A, dev_B, dev_C, Arow, Acol, Bcol);
        } else {
            // Run the basic version
            matrixMultiplyBasic<<<gridDim, blockDim>>>(dev_A, dev_B, dev_C, Arow, Acol, Bcol);
        }
        // Check for errors during kernel execution
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    
    // Synchronize to ensure all GPU operations complete
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Calculate elapsed time per operation
    double t = timer.lap_ms() / (double)(nacc);
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(C, dev_C, Crow * Ccol * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate performance metrics
    double flops = 2.0 * (double)Arow * (double)Acol * (double)Bcol; // Multiplications and additions
    double gflops = flops / (t * 1000000.0);
    
    // Calculate memory bandwidth (bytes transferred per second)
    // Reading A and B, writing C
    double dataTransferred = (Arow * Acol + Brow * Bcol + Crow * Ccol) * sizeof(float);
    double gbytes = dataTransferred / (t * 1000000.0);
    
    // Print results
    if (useOptimized) {
        printf("A %d x %d B %d x %d gpu time %.3f ms GFlops %.1f GBytes %.1f (Tiled)\n", 
               Arow, Acol, Brow, Bcol, t, gflops, gbytes);
    } else {
        printf("A %d x %d B %d x %d gpu time %.3f ms GFlops %.1f GBytes %.1f (Basic)\n", 
               Arow, Acol, Brow, Bcol, t, gflops, gbytes);
    }
    
    // Clean up
    CHECK_CUDA_ERROR(cudaFree(dev_A));
    CHECK_CUDA_ERROR(cudaFree(dev_B));
    CHECK_CUDA_ERROR(cudaFree(dev_C));
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}