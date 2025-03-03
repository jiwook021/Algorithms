/**
 * @file main.cu
 * @brief Driver for CUDA batch normalization layer demo.
 */

#include "BatchNormalization.cuh"
#include <iostream>
#include <vector>
#include <cstdlib>

int main() {
    const int BATCH_SIZE   = 4;
    const int FEATURE_SIZE = 8;
    const int SPATIAL_SIZE = 16;
    const int NUM_ELEMENTS = BATCH_SIZE * FEATURE_SIZE * SPATIAL_SIZE;

    // Initialize input
    std::vector<float> hInput(NUM_ELEMENTS);
    srand(42);
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        hInput[i] = (float)rand() / RAND_MAX;
    }

    // Initialize gamma=1, beta=0
    std::vector<float> hGamma(FEATURE_SIZE, 1.0f);
    std::vector<float> hBeta(FEATURE_SIZE, 0.0f);

    // Allocate device memory
    float *dInput, *dOutput, *dMean, *dVar, *dGamma, *dBeta;
    CUDA_CHECK(cudaMalloc(&dInput, NUM_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutput, NUM_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMean, FEATURE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dVar, FEATURE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dGamma, FEATURE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBeta, FEATURE_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dInput, hInput.data(), NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dGamma, hGamma.data(), FEATURE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBeta, hBeta.data(), FEATURE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (FEATURE_SIZE + 255) / 256;

    // Compute statistics
    ComputeMeanKernel<<<blocks, 256>>>(dInput, dMean, BATCH_SIZE, FEATURE_SIZE, SPATIAL_SIZE);
    ComputeVarianceKernel<<<blocks, 256>>>(dInput, dMean, dVar, BATCH_SIZE, FEATURE_SIZE, SPATIAL_SIZE);

    // Forward pass
    int fwdBlocks = (NUM_ELEMENTS + 255) / 256;
    BatchNormForwardKernel<<<fwdBlocks, 256>>>(dInput, dOutput, dMean, dVar, dGamma, dBeta,
                                                BN_EPSILON, BATCH_SIZE, FEATURE_SIZE, SPATIAL_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output
    std::vector<float> hOutput(NUM_ELEMENTS);
    CUDA_CHECK(cudaMemcpy(hOutput.data(), dOutput, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Sample output values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  Output[" << i << "] = " << hOutput[i] << std::endl;
    }
    std::cout << "Batch normalization completed successfully." << std::endl;

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));
    CUDA_CHECK(cudaFree(dMean));
    CUDA_CHECK(cudaFree(dVar));
    CUDA_CHECK(cudaFree(dGamma));
    CUDA_CHECK(cudaFree(dBeta));

    return 0;
}
