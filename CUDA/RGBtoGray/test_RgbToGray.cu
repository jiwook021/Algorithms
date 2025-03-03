/**
 * @file test_RgbToGray.cu
 * @brief Unit tests for the RgbToGray CUDA kernel.
 *
 * Tests the grayscale conversion logic with known input values.
 * Does not require OpenCV -- uses raw pixel buffers.
 */

#include "RgbToGray.cuh"
#include <cstdio>
#include <cmath>

static int testsPassed = 0;
static int testsFailed = 0;

#define ASSERT_EQ(actual, expected, msg)                                       \
    do {                                                                       \
        if ((actual) != (expected)) {                                          \
            fprintf(stderr, "[FAIL] %s: expected %d, got %d\n",               \
                    msg, (int)(expected), (int)(actual));                      \
            testsFailed++;                                                     \
        } else {                                                               \
            testsPassed++;                                                     \
        }                                                                      \
    } while (0)

void TestPureWhite() {
    // BGR = (255, 255, 255) -> Gray = 255
    unsigned char hInput[] = {255, 255, 255};
    unsigned char hOutput[1] = {0};

    unsigned char *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, 3));
    CHECK_CUDA(cudaMalloc(&dOut, 1));
    CHECK_CUDA(cudaMemcpy(dIn, hInput, 3, cudaMemcpyHostToDevice));

    GrayscaleKernel<<<1, 1>>>(dIn, dOut, 1, 1, 3);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hOutput, dOut, 1, cudaMemcpyDeviceToHost));

    // 0.114*255 + 0.587*255 + 0.299*255 = 255
    ASSERT_EQ(hOutput[0], 255, "PureWhite");
    printf("[PASSED] TestPureWhite\n");

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
}

void TestPureBlack() {
    unsigned char hInput[] = {0, 0, 0};
    unsigned char hOutput[1] = {99};

    unsigned char *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, 3));
    CHECK_CUDA(cudaMalloc(&dOut, 1));
    CHECK_CUDA(cudaMemcpy(dIn, hInput, 3, cudaMemcpyHostToDevice));

    GrayscaleKernel<<<1, 1>>>(dIn, dOut, 1, 1, 3);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hOutput, dOut, 1, cudaMemcpyDeviceToHost));

    ASSERT_EQ(hOutput[0], 0, "PureBlack");
    printf("[PASSED] TestPureBlack\n");

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
}

void TestPureGreen() {
    // BGR = (0, 255, 0) -> Gray = 0.587*255 = 149
    unsigned char hInput[] = {0, 255, 0};
    unsigned char hOutput[1] = {0};

    unsigned char *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, 3));
    CHECK_CUDA(cudaMalloc(&dOut, 1));
    CHECK_CUDA(cudaMemcpy(dIn, hInput, 3, cudaMemcpyHostToDevice));

    GrayscaleKernel<<<1, 1>>>(dIn, dOut, 1, 1, 3);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hOutput, dOut, 1, cudaMemcpyDeviceToHost));

    ASSERT_EQ(hOutput[0], 149, "PureGreen");
    printf("[PASSED] TestPureGreen\n");

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
}

void TestMultiplePixels() {
    // 2x1 image: pixel0 = white, pixel1 = black
    unsigned char hInput[] = {255, 255, 255, 0, 0, 0};
    unsigned char hOutput[2] = {0, 0};

    unsigned char *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, 6));
    CHECK_CUDA(cudaMalloc(&dOut, 2));
    CHECK_CUDA(cudaMemcpy(dIn, hInput, 6, cudaMemcpyHostToDevice));

    GrayscaleKernel<<<1, dim3(2, 1)>>>(dIn, dOut, 2, 1, 3);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hOutput, dOut, 2, cudaMemcpyDeviceToHost));

    ASSERT_EQ(hOutput[0], 255, "MultiPixel[0]");
    ASSERT_EQ(hOutput[1], 0, "MultiPixel[1]");
    printf("[PASSED] TestMultiplePixels\n");

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
}

int main() {
    TestPureWhite();
    TestPureBlack();
    TestPureGreen();
    TestMultiplePixels();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
