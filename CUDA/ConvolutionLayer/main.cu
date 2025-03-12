#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    if((call) != cudaSuccess) {                                             \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__;       \
        std::cerr << " code=" << cudaGetErrorString(call) << std::endl;     \
        exit(EXIT_FAILURE);                                                 \
    }

// CUDA kernel for 2D convolution
__global__ void conv2D(const float* input, const float* kernel, float* output,
                       int inputWidth, int inputHeight,
                       int kernelWidth, int kernelHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    if (x >= outputWidth || y >= outputHeight) return;

    float sum = 0.0f;
    for (int ky = 0; ky < kernelHeight; ky++) {
        for (int kx = 0; kx < kernelWidth; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            sum += input[iy * inputWidth + ix] * kernel[ky * kernelWidth + kx];
        }
    }
    output[y * outputWidth + x] = sum;
}

// Helper to run convolution on GPU
void runConvolution(const cv::Mat& inputImg, const std::vector<float>& kernel, cv::Mat& outputImg,
                    int kernelWidth, int kernelHeight)
{
    int inputWidth = inputImg.cols;
    int inputHeight = inputImg.rows;
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    size_t inputSize = inputWidth * inputHeight * sizeof(float);
    size_t kernelSize = kernelWidth * kernelHeight * sizeof(float);
    size_t outputSize = outputWidth * outputHeight * sizeof(float);

    // Allocate GPU memory
    float *d_input, *d_kernel, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelSize));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, inputImg.ptr<float>(), inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice));

    // Define grid and block size
    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    conv2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,
                                    inputWidth, inputHeight, kernelWidth, kernelHeight);
    CUDA_CHECK(cudaGetLastError());

    // Copy back result
    outputImg.create(outputHeight, outputWidth, CV_32FC1);
    CUDA_CHECK(cudaMemcpy(outputImg.ptr<float>(), d_output, outputSize, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));
}

int main()
{
    // Load input image as grayscale and convert to float
    cv::Mat inputImg = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    if (inputImg.empty()) {
        std::cerr << "Failed to load input.png!" << std::endl;
        return -1;
    }

    inputImg.convertTo(inputImg, CV_32FC1, 1.0 / 255.0);

    // Define a simple Sobel kernel (horizontal edge detection)
    std::vector<float> kernel = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    int kernelWidth = 3;
    int kernelHeight = 3;

    // Output image
    cv::Mat outputImg;

    // Run convolution
    runConvolution(inputImg, kernel, outputImg, kernelWidth, kernelHeight);

    // Normalize and convert back to 8-bit image
    cv::normalize(outputImg, outputImg, 0, 255, cv::NORM_MINMAX);
    outputImg.convertTo(outputImg, CV_8UC1);

    // Save the output image
    if (!cv::imwrite("output.png", outputImg)) {
        std::cerr << "Failed to save output.png!" << std::endl;
        return -1;
    }

    std::cout << "Convolution completed. Check output.png" << std::endl;
    return 0;
}
