#include <cuda.h>
#include <stdio.h>
#include <math.h>

// Include stb_image and stb_image_write implementations
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Kernel to convert unsigned char to float
__global__ void convert_uchar_to_float(unsigned char *d_image_uchar, float *d_image_float, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_image_float[idx] = (float)d_image_uchar[idx];
    }
}

// Kernel for parallel sum reduction per channel
__global__ void sum_reduction(float *d_image_float, float *d_partial_sums, int N, int C) {
    extern __shared__ float s_sums[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int pixels_per_thread = (N + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    int start_pixel = (bid * blockDim.x + tid) * pixels_per_thread;
    int end_pixel = min(start_pixel + pixels_per_thread, N);

    // Initialize local sums for each channel (max 3 channels assumed)
    float local_sums[3] = {0.0f, 0.0f, 0.0f};

    // Accumulate sums for assigned pixels
    for (int p = start_pixel; p < end_pixel; p++) {
        for (int c = 0; c < C; c++) {
            local_sums[c] += d_image_float[p * C + c];
        }
    }

    // Store local sums in shared memory
    for (int c = 0; c < C; c++) {
        s_sums[c * blockDim.x + tid] = local_sums[c];
    }
    __syncthreads();

    // Reduce sums within the block for each channel
    for (int c = 0; c < C; c++) {
        float *s_sum_c = s_sums + c * blockDim.x;
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sum_c[tid] += s_sum_c[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            d_partial_sums[bid * C + c] = s_sum_c[0];
        }
    }
}

// Kernel for sum of squared differences reduction per channel
__global__ void sum_of_squares_reduction(float *d_image_float, float *d_mean, float *d_partial_sums, int N, int C) {
    extern __shared__ float s_sums[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int pixels_per_thread = (N + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    int start_pixel = (bid * blockDim.x + tid) * pixels_per_thread;
    int end_pixel = min(start_pixel + pixels_per_thread, N);

    // Initialize local sums for each channel (max 3 channels assumed)
    float local_sums[3] = {0.0f, 0.0f, 0.0f};

    // Accumulate sum of squared differences
    for (int p = start_pixel; p < end_pixel; p++) {
        for (int c = 0; c < C; c++) {
            float val = d_image_float[p * C + c] - d_mean[c];
            local_sums[c] += val * val;
        }
    }

    // Store local sums in shared memory
    for (int c = 0; c < C; c++) {
        s_sums[c * blockDim.x + tid] = local_sums[c];
    }
    __syncthreads();

    // Reduce sums within the block for each channel
    for (int c = 0; c < C; c++) {
        float *s_sum_c = s_sums + c * blockDim.x;
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sum_c[tid] += s_sum_c[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            d_partial_sums[bid * C + c] = s_sum_c[0];
        }
    }
}

// Kernel to normalize the image
__global__ void normalize(float *d_image_float, float *d_mean, float *d_stddev, float *d_normalized, int size, int C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int channel = idx % C;
        // Add small epsilon to avoid division by zero
        d_normalized[idx] = (d_image_float[idx] - d_mean[channel]) / (d_stddev[channel] + 1e-5f);
    }
}

int main() {
    // Load the image
    int W, H, C;
    unsigned char *image = stbi_load("input.png", &W, &H, &C, 0);
    if (!image) {
        printf("Error: Could not load input.png\n");
        return 1;
    }
    int N = W * H;       // Number of pixels
    int size = N * C;    // Total number of elements (pixels * channels)

    // Allocate GPU memory
    unsigned char *d_image_uchar;
    float *d_image_float, *d_partial_sums, *d_mean, *d_stddev, *d_normalized;
    cudaMalloc(&d_image_uchar, size * sizeof(unsigned char));
    cudaMalloc(&d_image_float, size * sizeof(float));
    cudaMalloc(&d_normalized, size * sizeof(float));
    cudaMalloc(&d_mean, C * sizeof(float));
    cudaMalloc(&d_stddev, C * sizeof(float));

    // Copy image to GPU
    cudaMemcpy(d_image_uchar, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Convert image to float
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    convert_uchar_to_float<<<grid_size, block_size>>>(d_image_uchar, d_image_float, size);

    // Set number of blocks for reduction kernels
    int B = 32;
    cudaMalloc(&d_partial_sums, B * C * sizeof(float));
    int shared_mem_size = C * block_size * sizeof(float);

    // Step 1: Compute sum of pixel values per channel
    sum_reduction<<<B, block_size, shared_mem_size>>>(d_image_float, d_partial_sums, N, C);

    // Transfer partial sums to host and compute total sum
    float *partial_sums = new float[B * C];
    cudaMemcpy(partial_sums, d_partial_sums, B * C * sizeof(float), cudaMemcpyDeviceToHost);
    float total_sum[3] = {0.0f, 0.0f, 0.0f};
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            total_sum[c] += partial_sums[b * C + c];
        }
    }

    // Compute mean per channel
    float mean[3];
    for (int c = 0; c < C; c++) {
        mean[c] = total_sum[c] / N;
    }
    cudaMemcpy(d_mean, mean, C * sizeof(float), cudaMemcpyHostToDevice);

    // Step 2: Compute sum of squared differences per channel
    sum_of_squares_reduction<<<B, block_size, shared_mem_size>>>(d_image_float, d_mean, d_partial_sums, N, C);

    // Transfer partial sums to host and compute total sum of squares
    cudaMemcpy(partial_sums, d_partial_sums, B * C * sizeof(float), cudaMemcpyDeviceToHost);
    float total_sum_of_squares[3] = {0.0f, 0.0f, 0.0f};
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            total_sum_of_squares[c] += partial_sums[b * C + c];
        }
    }

    // Compute standard deviation per channel
    float stddev[3];
    for (int c = 0; c < C; c++) {
        float variance = total_sum_of_squares[c] / N;
        stddev[c] = sqrtf(variance);
    }
    cudaMemcpy(d_stddev, stddev, C * sizeof(float), cudaMemcpyHostToDevice);

    // Step 3: Normalize the image
    normalize<<<grid_size, block_size>>>(d_image_float, d_mean, d_stddev, d_normalized, size, C);

    // Optional: Transfer normalized data back to host for verification or saving
    /*
    float *normalized = new float[size];
    cudaMemcpy(normalized, d_normalized, size * sizeof(float), cudaMemcpyDeviceToHost);
    // For saving as an image, scale normalized values to 0-255, omitted here
    delete[] normalized;
    */

    // Clean up
    cudaFree(d_image_uchar);
    cudaFree(d_image_float);
    cudaFree(d_partial_sums);
    cudaFree(d_mean);
    cudaFree(d_stddev);
    cudaFree(d_normalized);
    delete[] partial_sums;
    stbi_image_free(image);

    printf("Image normalization completed successfully.\n");
    return 0;
}