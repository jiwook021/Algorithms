#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel for computing softmax across channels for each pixel
__global__ void softmax_kernel(float* input, float* output, int H, int W, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = H * W;

    if (idx < total_pixels) {
        int base = idx * C;

        // Find the maximum value for numerical stability
        float max_val = input[base];
        for (int k = 1; k < C; k++) {
            float val = input[base + k];
            if (val > max_val) max_val = val;
        }

        // Compute the sum of exponentials
        float sum = 0.0f;
        for (int k = 0; k < C; k++) {
            sum += expf(input[base + k] - max_val);
        }

        // Compute softmax values
        for (int k = 0; k < C; k++) {
            float exp_val = expf(input[base + k] - max_val);
            output[base + k] = exp_val / sum;
        }
    }
}

// CUDA kernel for computing argmax across channels for each pixel
__global__ void argmax_kernel(float* softmax_output, int* argmax_output, int H, int W, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = H * W;

    if (idx < total_pixels) {
        int base = idx * C;
        int max_idx = 0;
        float max_val = softmax_output[base];
        for (int k = 1; k < C; k++) {
            float val = softmax_output[base + k];
            if (val > max_val) {
                max_val = val;
                max_idx = k;
            }
        }
        argmax_output[idx] = max_idx;
    }
}

int main() {
    // **Step 1: Load the input image**
    std::cout << "Loading input image 'input.png'..." << std::endl;
    cv::Mat image = cv::imread("input.png", cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Error: Could not load 'input.png'. Please ensure it exists in the working directory." << std::endl;
        return 1;
    }
    int H = image.rows;       // Height
    int W = image.cols;       // Width
    int C = image.channels(); // Number of channels (classes)
    std::cout << "Image loaded: " << H << "x" << W << " with " << C << " channels." << std::endl;

    // **Step 2: Convert image to float**
    std::cout << "Converting image to float..." << std::endl;
    image.convertTo(image, CV_32F);

    // **Step 3: Allocate GPU memory**
    std::cout << "Allocating GPU memory..." << std::endl;
    float *d_input, *d_softmax_output;
    int *d_argmax_output;
    size_t size_float = H * W * C * sizeof(float);
    size_t size_int = H * W * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_input, size_float));
    CUDA_CHECK(cudaMalloc(&d_softmax_output, size_float));
    CUDA_CHECK(cudaMalloc(&d_argmax_output, size_int));

    // **Step 4: Copy image data to GPU**
    std::cout << "Copying image data to GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_input, image.data, size_float, cudaMemcpyHostToDevice));

    // **Step 5: Compute softmax on GPU**
    std::cout << "Computing softmax on GPU..." << std::endl;
    int block_size = 256;
    int num_blocks = (H * W + block_size - 1) / block_size;
    softmax_kernel<<<num_blocks, block_size>>>(d_input, d_softmax_output, H, W, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // **Step 6: Compute argmax on GPU**
    std::cout << "Computing argmax on GPU..." << std::endl;
    argmax_kernel<<<num_blocks, block_size>>>(d_softmax_output, d_argmax_output, H, W, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // **Step 7: Copy argmax results to host**
    std::cout << "Copying argmax results to host..." << std::endl;
    cv::Mat argmax_host(H, W, CV_32SC1);
    CUDA_CHECK(cudaMemcpy(argmax_host.data, d_argmax_output, size_int, cudaMemcpyDeviceToHost));

    // **Step 8: Create output image**
    std::cout << "Creating output image..." << std::endl;
    // Define a color palette (BGR format)
    std::vector<cv::Vec3b> palette = {
        cv::Vec3b(0, 0, 255),   // Blue for class 0
        cv::Vec3b(0, 255, 0),   // Green for class 1
        cv::Vec3b(255, 0, 0),   // Red for class 2
        cv::Vec3b(255, 255, 0), // Yellow for class 3
        cv::Vec3b(255, 0, 255), // Magenta for class 4
        cv::Vec3b(0, 255, 255)  // Cyan for class 5
    };
    if (C > static_cast<int>(palette.size())) {
        std::cout << "Note: Number of classes (" << C << ") exceeds palette size (" 
                  << palette.size() << "). Extra classes will be black." << std::endl;
    }

    cv::Mat output_image(H, W, CV_8UC3);
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int class_idx = argmax_host.at<int>(i, j);
            output_image.at<cv::Vec3b>(i, j) = (class_idx < static_cast<int>(palette.size())) 
                ? palette[class_idx] 
                : cv::Vec3b(0, 0, 0); // Black for undefined classes
        }
    }

    // **Step 9: Save the output image**
    std::cout << "Saving output image as 'output.png'..." << std::endl;
    cv::imwrite("output.png", output_image);
    std::cout << "Output image 'output.png' saved successfully." << std::endl;

    // **Step 10: Clean up**
    std::cout << "Freeing GPU memory..." << std::endl;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_softmax_output));
    CUDA_CHECK(cudaFree(d_argmax_output));

    return 0;
}