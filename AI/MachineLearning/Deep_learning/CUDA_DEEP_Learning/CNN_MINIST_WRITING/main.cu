#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    cudnnFilterDescriptor_t filterDesc;
    cudnnTensorDescriptor_t outputDesc;
    float *d_weights;    // Device weights
    float *d_dw;         // Device weight gradients
    float *d_biases;     // Device biases
    float *d_db;         // Device bias gradients
} ConvLayer;

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void initConvLayer(ConvLayer *layer, int inC, int outC, int kH, int kW) {
    cudnnStatus_t status;

    // Create filter descriptor
    status = cudnnCreateFilterDescriptor(&layer->filterDesc);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cudnnCreateFilterDescriptor failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }
    status = cudnnSetFilter4dDescriptor(layer->filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outC, inC, kH, kW);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cudnnSetFilter4dDescriptor failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    // Create output tensor descriptor (example dimensions)
    status = cudnnCreateTensorDescriptor(&layer->outputDesc);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cudnnCreateTensorDescriptor failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }
    int n = 1, c = outC, h = 10, w = 10; // Example output: batch=1, channels=outC, h=w=10
    status = cudnnSetTensor4dDescriptor(layer->outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cudnnSetTensor4dDescriptor failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    // Allocate device memory
    int filterSize = outC * inC * kH * kW;
    int biasSize = outC;
    checkCudaError(cudaMalloc(&layer->d_weights, filterSize * sizeof(float)), "cudaMalloc d_weights");
    checkCudaError(cudaMalloc(&layer->d_dw, filterSize * sizeof(float)), "cudaMalloc d_dw");
    checkCudaError(cudaMalloc(&layer->d_biases, biasSize * sizeof(float)), "cudaMalloc d_biases");
    checkCudaError(cudaMalloc(&layer->d_db, biasSize * sizeof(float)), "cudaMalloc d_db");
}

void updateConvWeights(ConvLayer *layer, float lr, cublasHandle_t cublas) {
    cudnnStatus_t status;
    cublasStatus_t cublasStatus;

    // Get filter dimensions
    cudnnDataType_t filterDataType;
    cudnnTensorFormat_t filterFormat;
    int k, c_in, filter_h, filter_w;
    status = cudnnGetFilter4dDescriptor(layer->filterDesc, &filterDataType, &filterFormat, &k, &c_in, &filter_h, &filter_w);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cudnnGetFilter4dDescriptor failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }
    int filterSize = k * c_in * filter_h * filter_w;

    // Update weights: w = w - lr * dw
    float alpha = -lr;
    cublasStatus = cublasSaxpy(cublas, filterSize, &alpha, layer->d_dw, 1, layer->d_weights, 1);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSaxpy failed for weights\n");
        exit(1);
    }

    // Get output tensor dimensions for biases
    cudnnDataType_t tensorDataType;
    int n, c_out, h_out, w_out, nStride, cStride, hStride, wStride;
    status = cudnnGetTensor4dDescriptor(layer->outputDesc, &tensorDataType, &n, &c_out, &h_out, &w_out, &nStride, &cStride, &hStride, &wStride);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cudnnGetTensor4dDescriptor failed: %s\n", cudnnGetErrorString(status));
        exit(1);
    }

    // Update biases: b = b - lr * db
    cublasStatus = cublasSaxpy(cublas, c_out, &alpha, layer->d_db, 1, layer->d_biases, 1);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSaxpy failed for biases\n");
        exit(1);
    }

    // Optional consistency check
    if (k != c_out) {
        printf("Mismatch: filter k=%d, tensor c_out=%d\n", k, c_out);
        exit(1);
    }
}

int main() {
    ConvLayer layer = {0};
    cublasHandle_t cublas;
    cublasStatus_t cublasStatus = cublasCreate(&cublas);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate failed\n");
        exit(1);
    }

    // Initialize layer: 3 input channels, 64 output channels, 3x3 kernel
    initConvLayer(&layer, 3, 64, 3, 3);

    // Update weights with learning rate 0.01
    updateConvWeights(&layer, 0.01f, cublas);

    // Cleanup
    cudnnDestroyFilterDescriptor(layer.filterDesc);
    cudnnDestroyTensorDescriptor(layer.outputDesc);
    cudaFree(layer.d_weights);
    cudaFree(layer.d_dw);
    cudaFree(layer.d_biases);
    cudaFree(layer.d_db);
    cublasDestroy(cublas);

    printf("Execution completed successfully\n");
    return 0;
}