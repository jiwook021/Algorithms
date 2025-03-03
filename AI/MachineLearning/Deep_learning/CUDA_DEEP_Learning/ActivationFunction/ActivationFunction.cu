/**
 * @file ActivationFunction.cu
 * @brief CUDA activation function kernel implementations.
 *
 * ReLU:      f(x) = max(0, x)
 * LeakyReLU: f(x) = x > 0 ? x : alpha * x
 * ELU:       f(x) = x > 0 ? x : alpha * (exp(x) - 1)
 * GELU:      f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Softmax:   f(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 */

#include "ActivationFunction.cuh"
#include <cfloat>
#include <algorithm>

__global__ void ReluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void LeakyReluKernel(const float* input, float* output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = input[idx] > 0.0f ? input[idx] : alpha * input[idx];
}

__global__ void EluKernel(const float* input, float* output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = input[idx] > 0.0f ? input[idx] : alpha * (expf(input[idx]) - 1.0f);
}

__global__ void GeluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float SQRT_2_OVER_PI = 0.7978845608028654f;
        const float COEFF = 0.044715f;
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * (x + COEFF * x * x * x)));
    }
}

__global__ void SoftmaxKernel(const float* input, float* output, int batchSize, int dim) {
    extern __shared__ float sharedMem[];
    int batchIdx = blockIdx.x;
    if (batchIdx >= batchSize) return;

    const float* batchInput  = input + batchIdx * dim;
    float*       batchOutput = output + batchIdx * dim;

    // Find max
    float maxVal = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        maxVal = fmaxf(maxVal, batchInput[i]);
    }
    sharedMem[threadIdx.x] = maxVal;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sharedMem[threadIdx.x] = fmaxf(sharedMem[threadIdx.x], sharedMem[threadIdx.x + stride]);
        __syncthreads();
    }
    maxVal = sharedMem[0];

    // Compute exp sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += expf(batchInput[i] - maxVal);
    }
    sharedMem[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        __syncthreads();
    }
    sum = sharedMem[0];

    // Normalize
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        batchOutput[i] = expf(batchInput[i] - maxVal) / sum;
    }
}

// --- Host Wrappers ---

void Relu(const float* dInput, float* dOutput, int size) {
    int blocks = (size + 255) / 256;
    ReluKernel<<<blocks, 256>>>(dInput, dOutput, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void LeakyRelu(const float* dInput, float* dOutput, int size, float alpha) {
    int blocks = (size + 255) / 256;
    LeakyReluKernel<<<blocks, 256>>>(dInput, dOutput, size, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Elu(const float* dInput, float* dOutput, int size, float alpha) {
    int blocks = (size + 255) / 256;
    EluKernel<<<blocks, 256>>>(dInput, dOutput, size, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Gelu(const float* dInput, float* dOutput, int size) {
    int blocks = (size + 255) / 256;
    GeluKernel<<<blocks, 256>>>(dInput, dOutput, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Softmax(const float* dInput, float* dOutput, int batchSize, int dim) {
    int blockSize = std::min(256, dim);
    SoftmaxKernel<<<batchSize, blockSize, blockSize * sizeof(float)>>>(dInput, dOutput, batchSize, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}
