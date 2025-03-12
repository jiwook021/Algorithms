#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * CUDA kernel for matrix multiplication
 * Each thread computes one element of the output matrix
 * 
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C = A * B
 * @param A_rows Number of rows in matrix A
 * @param A_cols Number of columns in matrix A (same as B_rows)
 * @param B_cols Number of columns in matrix B
 */
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, 
                                     int A_rows, int A_cols, int B_cols) {
    // Calculate global row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within matrix bounds
    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        
        // Compute dot product of row of A and column of B
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        
        // Store result in output matrix C
        C[row * B_cols + col] = sum;
    }
}

/**
 * Host function to perform matrix multiplication using CUDA
 * 
 * @param A Input matrix A as 1D array in row-major order
 * @param B Input matrix B as 1D array in row-major order
 * @param C Output matrix C = A * B as 1D array in row-major order
 * @param A_rows Number of rows in matrix A
 * @param A_cols Number of columns in matrix A (same as B_rows)
 * @param B_cols Number of columns in matrix B
 * @return True if operation was successful, false otherwise
 */
bool matrixMultiplyCuda(const float* A, const float* B, float* C,
                        int A_rows, int A_cols, int B_cols) {
    // Input validation
    if (A == nullptr || B == nullptr || C == nullptr) {
        std::cerr << "Error: Null matrix pointer provided" << std::endl;
        return false;
    }
    
    if (A_rows <= 0 || A_cols <= 0 || B_cols <= 0) {
        std::cerr << "Error: Invalid matrix dimensions" << std::endl;
        return false;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_rows * A_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, A_cols * B_cols * sizeof(float)));  // Note: B_rows = A_cols
    CUDA_CHECK(cudaMalloc(&d_C, A_rows * B_cols * sizeof(float)));
    
    // Copy input matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, A, A_rows * A_cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, A_cols * B_cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Define thread block and grid dimensions
    // Using a 16x16 thread block as a common choice for matrix operations
    dim3 blockSize(16, 16);
    dim3 gridSize((B_cols + blockSize.x - 1) / blockSize.x, 
                  (A_rows + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    
    // Check for kernel execution errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C, d_C, A_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return true;
}

/**
 * Helper function to print a matrix
 * 
 * @param matrix The matrix to print (1D array in row-major order)
 * @param rows Number of rows
 * @param cols Number of columns
 */
void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * CPU implementation of matrix multiplication for verification
 * 
 * @param A Input matrix A as 1D array in row-major order
 * @param B Input matrix B as 1D array in row-major order
 * @param C Output matrix C = A * B as 1D array in row-major order
 * @param A_rows Number of rows in matrix A
 * @param A_cols Number of columns in matrix A (same as B_rows)
 * @param B_cols Number of columns in matrix B
 */
void matrixMultiplyCpu(const float* A, const float* B, float* C,
                       int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; k++) {
                sum += A[i * A_cols + k] * B[k * B_cols + j];
            }
            C[i * B_cols + j] = sum;
        }
    }
}

int main() {
    // Define matrix dimensions
    const int A_rows = 2;
    const int A_cols = 3;
    const int B_rows = A_cols;  // Must equal A_cols for multiplication
    const int B_cols = 2;
    const int C_rows = A_rows;
    const int C_cols = B_cols;
    
    // Host matrices
    float A[A_rows * A_cols] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float B[B_rows * B_cols] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float C[C_rows * C_cols] = {0.0f};  // Result matrix
    float C_cpu[C_rows * C_cols] = {0.0f};  // CPU result for verification
    
    // Print input matrices
    std::cout << "Matrix A:" << std::endl;
    printMatrix(A, A_rows, A_cols);
    
    std::cout << "Matrix B:" << std::endl;
    printMatrix(B, B_rows, B_cols);
    
    // Perform matrix multiplication using CUDA
    std::cout << "Performing matrix multiplication using CUDA..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    bool success = matrixMultiplyCuda(A, B, C, A_rows, A_cols, B_cols);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (success) {
        std::chrono::duration<double, std::milli> gpu_time = end - start;
        std::cout << "CUDA Execution Time: " << gpu_time.count() << " ms" << std::endl;
        
        // Print result matrix
        std::cout << "Result Matrix C (CUDA):" << std::endl;
        printMatrix(C, C_rows, C_cols);
        
        // Verify result with CPU calculation
        std::cout << "Verifying result with CPU calculation..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        matrixMultiplyCpu(A, B, C_cpu, A_rows, A_cols, B_cols);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = end - start;
        
        std::cout << "CPU Execution Time: " << cpu_time.count() << " ms" << std::endl;
        
        std::cout << "Result Matrix C (CPU):" << std::endl;
        printMatrix(C_cpu, C_rows, C_cols);
        
        // Compare results
        bool match = true;
        for (int i = 0; i < C_rows * C_cols; i++) {
            if (std::abs(C[i] - C_cpu[i]) > 1e-5f) {
                match = false;
                break;
            }
        }
        
        if (match) {
            std::cout << "Verification PASSED: CUDA and CPU results match." << std::endl;
        } else {
            std::cout << "Verification FAILED: CUDA and CPU results differ." << std::endl;
        }
    } else {
        std::cout << "Matrix multiplication failed." << std::endl;
    }
    
    return 0;
}