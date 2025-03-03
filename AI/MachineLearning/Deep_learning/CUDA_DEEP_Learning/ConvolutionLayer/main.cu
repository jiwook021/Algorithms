/**
 * @file main.cu
 * @brief Driver for CUDA 2D convolution on image data.
 *
 * Dependencies: OpenCV
 */

#include "ConvolutionLayer.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    cv::Mat inputImg = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    if (inputImg.empty()) {
        std::cerr << "Failed to load input.png!" << std::endl;
        return -1;
    }

    inputImg.convertTo(inputImg, CV_32FC1, 1.0 / 255.0);

    // Sobel kernel (horizontal edge detection)
    std::vector<float> kernel = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    int kernelWidth = 3, kernelHeight = 3;
    int inputWidth = inputImg.cols, inputHeight = inputImg.rows;
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    float *dInput, *dKernel, *dOutput;
    CUDA_CHECK(cudaMalloc(&dInput, inputWidth * inputHeight * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dKernel, 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutput, outputWidth * outputHeight * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dInput, inputImg.ptr<float>(), inputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKernel, kernel.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + 15) / 16, (outputHeight + 15) / 16);
    Conv2DKernel<<<gridSize, blockSize>>>(dInput, dKernel, dOutput, inputWidth, inputHeight, kernelWidth, kernelHeight);
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::Mat outputImg(outputHeight, outputWidth, CV_32FC1);
    CUDA_CHECK(cudaMemcpy(outputImg.ptr<float>(), dOutput, outputWidth * outputHeight * sizeof(float), cudaMemcpyDeviceToHost));

    cv::normalize(outputImg, outputImg, 0, 255, cv::NORM_MINMAX);
    outputImg.convertTo(outputImg, CV_8UC1);
    cv::imwrite("output.png", outputImg);

    std::cout << "Convolution completed. Check output.png" << std::endl;

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dKernel));
    CUDA_CHECK(cudaFree(dOutput));

    return 0;
}
