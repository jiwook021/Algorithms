/**
 * @file main.cu
 * @brief Driver for CUDA parallel softmax + argmax on image data.
 *
 * Dependencies: OpenCV
 */

#include "ParallelSoftmax.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Loading input image 'input.png'..." << std::endl;
    cv::Mat image = cv::imread("input.png", cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Error: Could not load 'input.png'." << std::endl;
        return 1;
    }

    int h = image.rows;
    int w = image.cols;
    int c = image.channels();
    std::cout << "Image: " << h << "x" << w << " with " << c << " channels." << std::endl;

    image.convertTo(image, CV_32F);

    float *dInput, *dSoftmax;
    int *dArgmax;
    size_t sizeFloat = h * w * c * sizeof(float);
    size_t sizeInt   = h * w * sizeof(int);
    CUDA_CHECK(cudaMalloc(&dInput, sizeFloat));
    CUDA_CHECK(cudaMalloc(&dSoftmax, sizeFloat));
    CUDA_CHECK(cudaMalloc(&dArgmax, sizeInt));
    CUDA_CHECK(cudaMemcpy(dInput, image.data, sizeFloat, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (h * w + blockSize - 1) / blockSize;

    SoftmaxKernel<<<numBlocks, blockSize>>>(dInput, dSoftmax, h, w, c);
    CUDA_CHECK(cudaDeviceSynchronize());

    ArgmaxKernel<<<numBlocks, blockSize>>>(dSoftmax, dArgmax, h, w, c);
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::Mat argmaxHost(h, w, CV_32SC1);
    CUDA_CHECK(cudaMemcpy(argmaxHost.data, dArgmax, sizeInt, cudaMemcpyDeviceToHost));

    // Color-map output
    std::vector<cv::Vec3b> palette = {
        cv::Vec3b(0, 0, 255),   cv::Vec3b(0, 255, 0),   cv::Vec3b(255, 0, 0),
        cv::Vec3b(255, 255, 0), cv::Vec3b(255, 0, 255),  cv::Vec3b(0, 255, 255)
    };

    cv::Mat outputImage(h, w, CV_8UC3);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            int classIdx = argmaxHost.at<int>(i, j);
            outputImage.at<cv::Vec3b>(i, j) = (classIdx < (int)palette.size())
                ? palette[classIdx] : cv::Vec3b(0, 0, 0);
        }
    }

    cv::imwrite("output.png", outputImage);
    std::cout << "Output saved as 'output.png'." << std::endl;

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dSoftmax));
    CUDA_CHECK(cudaFree(dArgmax));

    return 0;
}
