#ifndef ACTIVATION_FUNCTION_CUH
#define ACTIVATION_FUNCTION_CUH

/**
 * @file ActivationFunction.cuh
 * @brief Header for CUDA activation function kernels.
 *
 * Implements: ReLU, LeakyReLU, ELU, GELU, and Softmax.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(Call)                                                      \
    do {                                                                      \
        cudaError_t Err = Call;                                               \
        if (Err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(Err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// --- CUDA Kernels ---

__global__ void ReluKernel(const float* input, float* output, int size);
__global__ void LeakyReluKernel(const float* input, float* output, int size, float alpha);
__global__ void EluKernel(const float* input, float* output, int size, float alpha);
__global__ void GeluKernel(const float* input, float* output, int size);
__global__ void SoftmaxKernel(const float* input, float* output, int batchSize, int dim);

// --- Host Wrappers ---

void Relu(const float* dInput, float* dOutput, int size);
void LeakyRelu(const float* dInput, float* dOutput, int size, float alpha = 0.01f);
void Elu(const float* dInput, float* dOutput, int size, float alpha = 1.0f);
void Gelu(const float* dInput, float* dOutput, int size);
void Softmax(const float* dInput, float* dOutput, int batchSize, int dim);

#endif // ACTIVATION_FUNCTION_CUH
