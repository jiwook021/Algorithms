/**
 * @file main.cu
 * @brief Driver for MaxPooling — runs a quick demo with a 4x4 input.
 */

#include "MaxPooling.cuh"
#include <iostream>
#include <vector>

int main() {
    const int INPUT_WIDTH  = 4;
    const int INPUT_HEIGHT = 4;
    const int POOL_WIDTH   = 2;
    const int POOL_HEIGHT  = 2;

    std::vector<float> input = {
        1, 2, 5, 4,
        5, 6, 7, 8,
        3, 2, 1, 0,
        1, 2, 3, 4
    };

    std::vector<float> output;
    MaxPooling(input, output, INPUT_WIDTH, INPUT_HEIGHT, POOL_WIDTH, POOL_HEIGHT);

    int outputWidth  = (INPUT_WIDTH  + POOL_WIDTH  - 1) / POOL_WIDTH;
    int outputHeight = (INPUT_HEIGHT + POOL_HEIGHT - 1) / POOL_HEIGHT;

    std::cout << "Max-pooled output (" << outputWidth << "x" << outputHeight << "):" << std::endl;
    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            std::cout << output[y * outputWidth + x] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
