#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// Grayscale CUDA kernel
__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output,
                                int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int colorIdx = idx * channels;

        float blue  = input[colorIdx];
        float green = input[colorIdx + 1];
        float red   = input[colorIdx + 2];

        output[idx] = (unsigned char)(0.114f * blue + 0.587f * green + 0.299f * red);
    }
}

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

    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    size_t colorSize = width * height * channels * sizeof(unsigned char);
    size_t grayscaleSize = width * height * sizeof(unsigned char);

    unsigned char *d_input = NULL;
    unsigned char *d_output = NULL;

    cudaMalloc((void**)&d_input, colorSize);
    cudaMalloc((void**)&d_output, grayscaleSize);

    cudaMemcpy(d_input, inputImage.data, colorSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return 1;
    }

    cv::Mat grayscaleImage(height, width, CV_8UC1);
    cudaMemcpy(grayscaleImage.data, d_output, grayscaleSize, cudaMemcpyDeviceToHost);

    cv::imwrite("grayscale_output.jpg", grayscaleImage);
    printf("Grayscale image saved as grayscale_output.jpg\n");

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
