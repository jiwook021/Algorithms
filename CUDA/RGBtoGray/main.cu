/**
 * @file main.cu
 * @brief Driver for CUDA RGB-to-Grayscale conversion.
 *
 * Usage: ./main <input_image>
 * Output: grayscale_output.jpg
 *
 * Dependencies: OpenCV
 */

#include "RgbToGray.cuh"
#include <cstdio>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <input_image>\n", argv[0]);
        return 1;
    }

    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        printf("Error: Could not read image %s\n", argv[1]);
        return 1;
    }

    int width    = inputImage.cols;
    int height   = inputImage.rows;
    int channels = inputImage.channels();

    size_t colorSize     = width * height * channels * sizeof(unsigned char);
    size_t grayscaleSize = width * height * sizeof(unsigned char);

    unsigned char *dInput  = NULL;
    unsigned char *dOutput = NULL;
    CHECK_CUDA(cudaMalloc(&dInput, colorSize));
    CHECK_CUDA(cudaMalloc(&dOutput, grayscaleSize));
    CHECK_CUDA(cudaMemcpy(dInput, inputImage.data, colorSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width  + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    GrayscaleKernel<<<gridSize, blockSize>>>(dInput, dOutput, width, height, channels);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cv::Mat grayscaleImage(height, width, CV_8UC1);
    CHECK_CUDA(cudaMemcpy(grayscaleImage.data, dOutput, grayscaleSize, cudaMemcpyDeviceToHost));

    cv::imwrite("grayscale_output.jpg", grayscaleImage);
    printf("Grayscale image saved as grayscale_output.jpg\n");

    CHECK_CUDA(cudaFree(dInput));
    CHECK_CUDA(cudaFree(dOutput));

    return 0;
}
