/**
 * @file main.cu
 * @brief Driver for CUDA activation function benchmarks.
 */

#include "ActivationFunction.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

int main() {
    const int SIZE = 1 << 20;  // 1M elements
    const int TEST_SIZE = 5;

    std::vector<float> hInput(SIZE);
    std::vector<float> hOutput(SIZE);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < SIZE; i++) hInput[i] = dist(gen);

    float *dInput, *dOutput;
    CUDA_CHECK(cudaMalloc(&dInput, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutput, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dInput, hInput.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Test each activation with small input
    std::vector<float> testInput = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> testOutput(TEST_SIZE);
    CUDA_CHECK(cudaMemcpy(dInput, testInput.data(), TEST_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    auto printTest = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(testOutput.data(), dOutput, TEST_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << name << ": ";
        for (auto v : testOutput) std::cout << v << " ";
        std::cout << std::endl;
    };

    Relu(dInput, dOutput, TEST_SIZE);
    printTest("ReLU");

    LeakyRelu(dInput, dOutput, TEST_SIZE, 0.01f);
    printTest("LeakyReLU");

    Elu(dInput, dOutput, TEST_SIZE, 1.0f);
    printTest("ELU");

    Gelu(dInput, dOutput, TEST_SIZE);
    printTest("GELU");

    Softmax(dInput, dOutput, 1, TEST_SIZE);
    printTest("Softmax");

    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dOutput));

    std::cout << "All activation function demos completed." << std::endl;
    return 0;
}
