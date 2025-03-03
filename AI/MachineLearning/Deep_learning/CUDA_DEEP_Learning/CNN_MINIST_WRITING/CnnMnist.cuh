#ifndef CNN_MNIST_CUH
#define CNN_MNIST_CUH

/**
 * @file CnnMnist.cuh
 * @brief Header for CNN MNIST digit recognition using cuDNN and cuBLAS.
 *
 * Defines the ConvLayer structure and helper functions for
 * weight initialization and gradient-based weight updates.
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

typedef struct {
    cudnnFilterDescriptor_t FilterDesc;
    cudnnTensorDescriptor_t OutputDesc;
    float* DWeights;    // Device weights
    float* DDw;         // Device weight gradients
    float* DBiases;     // Device biases
    float* DDb;         // Device bias gradients
} ConvLayer;

void CheckCudaError(cudaError_t err, const char* msg);
void InitConvLayer(ConvLayer* layer, int inC, int outC, int kH, int kW);
void UpdateConvWeights(ConvLayer* layer, float lr, cublasHandle_t cublas);

#endif // CNN_MNIST_CUH
