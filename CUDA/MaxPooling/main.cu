// Parallel Max-Pooling Kernel Implementation in CUDA
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cfloat>  // <-- Added this header for FLT_MAX

// CUDA kernel for max-pooling
__global__ void maxPoolingKernel(const float* input, float* output, 
                                int inputWidth, int inputHeight,
                                int poolWidth, int poolHeight,
                                int outputWidth, int outputHeight) {

    // Calculate output element (x, y)
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outX >= outputWidth || outY >= outputHeight)
        return;

    // Define pooling region
    int startX = outX * poolWidth;
    int startY = outY * poolHeight;

    float maxVal = -FLT_MAX;

    // Iterate over pooling region
    for (int i = 0; i < poolHeight; ++i) {
        for (int j = 0; j < poolWidth; ++j) {
            int curX = startX + j;
            int curY = startY + i;

            if (curX < inputWidth && curY < inputHeight) {
                float val = input[curY * inputWidth + curX];
                maxVal = fmaxf(maxVal, val);
            }
        }
    }

    output[outY * outputWidth + outX] = maxVal;
}

// Host function to perform max-pooling
void maxPooling(const std::vector<float>& input, std::vector<float>& output,
                int inputWidth, int inputHeight,
                int poolWidth, int poolHeight) {

    assert(input.size() == inputWidth * inputHeight);

    int outputWidth = (inputWidth + poolWidth - 1) / poolWidth;
    int outputHeight = (inputHeight + poolHeight - 1) / poolHeight;
    output.resize(outputWidth * outputHeight);

    float *d_input, *d_output;

    size_t inputBytes = input.size() * sizeof(float);
    size_t outputBytes = output.size() * sizeof(float);

    cudaMalloc(&d_input, inputBytes);
    cudaMalloc(&d_output, outputBytes);

    cudaMemcpy(d_input, input.data(), inputBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);

    maxPoolingKernel<<<gridSize, blockSize>>>(d_input, d_output,
                                              inputWidth, inputHeight,
                                              poolWidth, poolHeight,
                                              outputWidth, outputHeight);

    cudaMemcpy(output.data(), d_output, outputBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// Test function
void testMaxPooling() {
    const int inputWidth = 4;
    const int inputHeight = 4;
    const int poolWidth = 2;
    const int poolHeight = 2;

    std::vector<float> input = {
        1, 2, 5, 4,
        5, 6, 7, 8,
        3, 2, 1, 0,
        1, 2, 3, 4
    };

    std::vector<float> output;

    maxPooling(input, output, inputWidth, inputHeight, poolWidth, poolHeight);

    // Expected Output: [6, 8, 3, 4]
    std::vector<float> expectedOutput = {6, 8, 3, 4};

    std::cout << "Max Pooling Output:\n";
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << output[i] << " ";
        assert(output[i] == expectedOutput[i]);
    }
    std::cout << "\nTest Passed!" << std::endl;
}

int main() {
    testMaxPooling();
    return 0;
}
