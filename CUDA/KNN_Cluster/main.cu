#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Include stb_image.h for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// CUDA kernel to compute squared Euclidean distances
__global__ void compute_distances(float* train, float* query, float* distances, int N, int D) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < N) {
        float dist = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = train[t * D + d] - query[d];
            dist += diff * diff;
        }
        distances[t] = dist;
    }
}

// CUDA kernel to count votes for each class among k nearest neighbors
__global__ void count_labels(int* labels, int* indices, int* counters, int k) {
    int t = threadIdx.x;
    if (t < k) {
        int idx = indices[t];
        int label = labels[idx];
        atomicAdd(&counters[label], 1);
    }
}

int main() {
    // Define constants
    const int N = 1000;  // Number of training samples
    const int D = 3072;  // Feature dimension (32x32x3)
    const int C = 10;    // Number of classes
    const int k = 5;     // Number of nearest neighbors

    // --- Step 1: Generate synthetic training data ---
    float* train_features = new float[N * D];
    int* labels = new int[N];
    std::srand(std::time(0));
    for (int i = 0; i < N * D; i++) {
        train_features[i] = static_cast<float>(std::rand() % 256); // Mimic pixel values (0-255)
    }
    for (int i = 0; i < N; i++) {
        labels[i] = std::rand() % C; // Random labels from 0 to C-1
    }

    // --- Step 2: Load and process input.png ---
    int width, height, channels;
    unsigned char* img = stbi_load("input.png", &width, &height, &channels, 3);
    if (img == nullptr || width != 32 || height != 32 || channels != 3) {
        std::cerr << "Error: input.png must be a 32x32 RGB image" << std::endl;
        return 1;
    }
    float* query = new float[D];
    for (int i = 0; i < D; i++) {
        query[i] = static_cast<float>(img[i]); // Convert pixel values to float
    }
    stbi_image_free(img);

    // --- Step 3: Allocate GPU memory ---
    float *d_train, *d_query, *d_distances;
    int *d_labels, *d_indices, *d_counters;
    cudaMalloc(&d_train, N * D * sizeof(float));
    cudaMalloc(&d_query, D * sizeof(float));
    cudaMalloc(&d_distances, N * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_indices, N * sizeof(int));
    cudaMalloc(&d_counters, C * sizeof(int));

    // --- Step 4: Copy data to GPU ---
    cudaMemcpy(d_train, train_features, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, D * sizeof(float), cudaMemcpyHostToDevice);

    // --- Step 5: Compute distances in parallel ---
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    compute_distances<<<num_blocks, block_size>>>(d_train, d_query, d_distances, N, D);
    cudaDeviceSynchronize();

    // --- Step 6: Sort distances using Thrust ---
    thrust::device_ptr<int> idx_ptr(d_indices);
    thrust::device_ptr<float> dist_ptr(d_distances);
    thrust::sequence(idx_ptr, idx_ptr + N); // Initialize indices: 0, 1, ..., N-1
    thrust::sort_by_key(dist_ptr, dist_ptr + N, idx_ptr); // Sort distances with indices

    // --- Step 7: Count votes for k nearest neighbors ---
    cudaMemset(d_counters, 0, C * sizeof(int)); // Reset counters
    count_labels<<<1, k>>>(d_labels, d_indices, d_counters, k);
    cudaDeviceSynchronize();

    // --- Step 8: Copy counters to host and find majority class ---
    int* h_counters = new int[C];
    cudaMemcpy(h_counters, d_counters, C * sizeof(int), cudaMemcpyDeviceToHost);
    int max_count = 0;
    int predicted_class = -1;
    for (int c = 0; c < C; c++) {
        if (h_counters[c] > max_count) {
            max_count = h_counters[c];
            predicted_class = c;
        }
    }
    std::cout << "Predicted class: " << predicted_class << std::endl;

    // --- Step 9: Clean up ---
    delete[] train_features;
    delete[] labels;
    delete[] query;
    delete[] h_counters;
    cudaFree(d_train);
    cudaFree(d_query);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_indices);
    cudaFree(d_counters);

    return 0;
}