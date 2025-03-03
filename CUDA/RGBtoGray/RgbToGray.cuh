#ifndef RGB_TO_GRAY_CUH
#define RGB_TO_GRAY_CUH

/**
 * @file RgbToGray.cuh
 * @brief Header for CUDA RGB-to-Grayscale conversion kernel.
 *
 * Applies ITU-R BT.601 luminance formula:
 *   Gray = 0.114 * Blue + 0.587 * Green + 0.299 * Red
 */

#include <cuda_runtime.h>

#define CHECK_CUDA(Call)                                                      \
    do {                                                                      \
        cudaError_t Err = Call;                                               \
        if (Err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(Err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/**
 * @brief Converts one pixel from BGR to grayscale using BT.601 weights.
 *
 * @param input    Device pointer to BGR image (interleaved).
 * @param output   Device pointer to grayscale output (one byte per pixel).
 * @param width    Image width in pixels.
 * @param height   Image height in pixels.
 * @param channels Number of input channels (expected: 3).
 */
__global__ void GrayscaleKernel(const unsigned char* input, unsigned char* output,
                                int width, int height, int channels);

#endif // RGB_TO_GRAY_CUH
