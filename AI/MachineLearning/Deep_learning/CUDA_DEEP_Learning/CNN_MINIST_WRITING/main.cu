/**
 * @file main.cu
 * @brief Driver for CNN MNIST digit recognition layer demo.
 *
 * Dependencies: cuDNN, cuBLAS
 */

#include "CnnMnist.cuh"

int main() {
    ConvLayer layer = {};
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    // Initialize layer: 3 input channels, 64 output channels, 3x3 kernel
    InitConvLayer(&layer, 3, 64, 3, 3);

    // Update weights with learning rate 0.01
    UpdateConvWeights(&layer, 0.01f, cublas);

    // Cleanup
    cudnnDestroyFilterDescriptor(layer.FilterDesc);
    cudnnDestroyTensorDescriptor(layer.OutputDesc);
    cudaFree(layer.DWeights);
    cudaFree(layer.DDw);
    cudaFree(layer.DBiases);
    cudaFree(layer.DDb);
    cublasDestroy(cublas);

    printf("CNN MNIST layer demo completed successfully.\n");
    return 0;
}
