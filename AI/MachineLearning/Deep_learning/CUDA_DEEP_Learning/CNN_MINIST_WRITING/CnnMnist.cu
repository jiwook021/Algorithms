/**
 * @file CnnMnist.cu
 * @brief CNN MNIST implementation using cuDNN and cuBLAS.
 *
 * Provides conv layer initialization and SGD weight update functions.
 */

#include "CnnMnist.cuh"

void CheckCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void InitConvLayer(ConvLayer* layer, int inC, int outC, int kH, int kW) {
    cudnnCreateFilterDescriptor(&layer->FilterDesc);
    cudnnSetFilter4dDescriptor(layer->FilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC, inC, kH, kW);

    cudnnCreateTensorDescriptor(&layer->OutputDesc);
    int n = 1, c = outC, h = 10, w = 10;
    cudnnSetTensor4dDescriptor(layer->OutputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);

    int filterSize = outC * inC * kH * kW;
    int biasSize = outC;
    CheckCudaError(cudaMalloc(&layer->DWeights, filterSize * sizeof(float)), "cudaMalloc weights");
    CheckCudaError(cudaMalloc(&layer->DDw, filterSize * sizeof(float)), "cudaMalloc dw");
    CheckCudaError(cudaMalloc(&layer->DBiases, biasSize * sizeof(float)), "cudaMalloc biases");
    CheckCudaError(cudaMalloc(&layer->DDb, biasSize * sizeof(float)), "cudaMalloc db");
}

void UpdateConvWeights(ConvLayer* layer, float lr, cublasHandle_t cublas) {
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    int k, cIn, filterH, filterW;
    cudnnGetFilter4dDescriptor(layer->FilterDesc, &dataType, &format, &k, &cIn, &filterH, &filterW);
    int filterSize = k * cIn * filterH * filterW;

    float alpha = -lr;
    cublasSaxpy(cublas, filterSize, &alpha, layer->DDw, 1, layer->DWeights, 1);

    cudnnDataType_t tensorType;
    int n, cOut, hOut, wOut, nS, cS, hS, wS;
    cudnnGetTensor4dDescriptor(layer->OutputDesc, &tensorType, &n, &cOut, &hOut, &wOut, &nS, &cS, &hS, &wS);
    cublasSaxpy(cublas, cOut, &alpha, layer->DDb, 1, layer->DBiases, 1);
}
