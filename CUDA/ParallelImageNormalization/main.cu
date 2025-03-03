/**
 * @file main.cu
 * @brief Driver for CUDA parallel image normalization.
 *
 * Loads input.png, normalizes per-channel, and reports results.
 *
 * Dependencies: stb_image.h, stb_image_write.h
 */

#include "ParallelImageNorm.cuh"
#include <cstdio>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    int w, h, c;
    unsigned char* image = stbi_load("input.png", &w, &h, &c, 0);
    if (!image) {
        printf("Error: Could not load input.png\n");
        return 1;
    }
    int n    = w * h;
    int size = n * c;

    unsigned char* dImageUchar;
    float *dImageFloat, *dPartialSums, *dMean, *dStddev, *dNormalized;
    cudaMalloc(&dImageUchar, size * sizeof(unsigned char));
    cudaMalloc(&dImageFloat, size * sizeof(float));
    cudaMalloc(&dNormalized, size * sizeof(float));
    cudaMalloc(&dMean, c * sizeof(float));
    cudaMalloc(&dStddev, c * sizeof(float));

    cudaMemcpy(dImageUchar, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize  = (size + blockSize - 1) / blockSize;
    ConvertUcharToFloat<<<gridSize, blockSize>>>(dImageUchar, dImageFloat, size);

    int numBlocks = 32;
    cudaMalloc(&dPartialSums, numBlocks * c * sizeof(float));
    int sharedMemSize = c * blockSize * sizeof(float);

    // Compute mean
    SumReductionKernel<<<numBlocks, blockSize, sharedMemSize>>>(dImageFloat, dPartialSums, n, c);
    float* partialSums = new float[numBlocks * c];
    cudaMemcpy(partialSums, dPartialSums, numBlocks * c * sizeof(float), cudaMemcpyDeviceToHost);

    float totalSum[3] = {0.0f, 0.0f, 0.0f};
    for (int b = 0; b < numBlocks; b++)
        for (int ch = 0; ch < c; ch++)
            totalSum[ch] += partialSums[b * c + ch];

    float mean[3];
    for (int ch = 0; ch < c; ch++) mean[ch] = totalSum[ch] / n;
    cudaMemcpy(dMean, mean, c * sizeof(float), cudaMemcpyHostToDevice);

    // Compute stddev
    SumOfSquaresReductionKernel<<<numBlocks, blockSize, sharedMemSize>>>(dImageFloat, dMean, dPartialSums, n, c);
    cudaMemcpy(partialSums, dPartialSums, numBlocks * c * sizeof(float), cudaMemcpyDeviceToHost);

    float totalSumSq[3] = {0.0f, 0.0f, 0.0f};
    for (int b = 0; b < numBlocks; b++)
        for (int ch = 0; ch < c; ch++)
            totalSumSq[ch] += partialSums[b * c + ch];

    float stddev[3];
    for (int ch = 0; ch < c; ch++) stddev[ch] = sqrtf(totalSumSq[ch] / n);
    cudaMemcpy(dStddev, stddev, c * sizeof(float), cudaMemcpyHostToDevice);

    // Normalize
    NormalizeKernel<<<gridSize, blockSize>>>(dImageFloat, dMean, dStddev, dNormalized, size, c);

    // Cleanup
    cudaFree(dImageUchar);
    cudaFree(dImageFloat);
    cudaFree(dPartialSums);
    cudaFree(dMean);
    cudaFree(dStddev);
    cudaFree(dNormalized);
    delete[] partialSums;
    stbi_image_free(image);

    printf("Image normalization completed successfully.\n");
    printf("Mean:   [%.2f, %.2f, %.2f]\n", mean[0], mean[1], mean[2]);
    printf("Stddev: [%.2f, %.2f, %.2f]\n", stddev[0], stddev[1], stddev[2]);

    return 0;
}
