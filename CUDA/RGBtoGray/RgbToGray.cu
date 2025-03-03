/**
 * @file RgbToGray.cu
 * @brief CUDA RGB-to-Grayscale kernel implementation.
 *
 * Each thread processes one pixel with BT.601 luminance weights.
 *
 * Complexity:
 *   - Time:  O(1) per thread, O(W*H) total
 *   - Space: O(W*H*3) input + O(W*H) output on device
 */

#include "RgbToGray.cuh"

__global__ void GrayscaleKernel(const unsigned char* input, unsigned char* output,
                                int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx      = y * width + x;
        int colorIdx = idx * channels;

        float blue  = input[colorIdx];
        float green = input[colorIdx + 1];
        float red   = input[colorIdx + 2];

        output[idx] = (unsigned char)(0.114f * blue + 0.587f * green + 0.299f * red);
    }
}
