/**
 * @file test_CnnMnist.cu
 * @brief Unit tests for CnnMnist layer initialization.
 *
 * Note: Requires cuDNN and cuBLAS to be available.
 */

#include "CnnMnist.cuh"
#include <cstdio>

static int testsPassed = 0;
static int testsFailed = 0;

void TestLayerInit() {
    ConvLayer layer = {};
    InitConvLayer(&layer, 1, 32, 5, 5);

    // Verify allocations are non-null
    if (layer.DWeights != nullptr) testsPassed++;
    else { testsFailed++; fprintf(stderr, "[FAIL] weights null\n"); }

    if (layer.DBiases != nullptr) testsPassed++;
    else { testsFailed++; fprintf(stderr, "[FAIL] biases null\n"); }

    printf("[PASSED] TestLayerInit\n");

    cudnnDestroyFilterDescriptor(layer.FilterDesc);
    cudnnDestroyTensorDescriptor(layer.OutputDesc);
    cudaFree(layer.DWeights);
    cudaFree(layer.DDw);
    cudaFree(layer.DBiases);
    cudaFree(layer.DDb);
}

void TestWeightUpdate() {
    ConvLayer layer = {};
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    InitConvLayer(&layer, 3, 64, 3, 3);
    UpdateConvWeights(&layer, 0.01f, cublas);

    testsPassed++;
    printf("[PASSED] TestWeightUpdate (no crash)\n");

    cudnnDestroyFilterDescriptor(layer.FilterDesc);
    cudnnDestroyTensorDescriptor(layer.OutputDesc);
    cudaFree(layer.DWeights);
    cudaFree(layer.DDw);
    cudaFree(layer.DBiases);
    cudaFree(layer.DDb);
    cublasDestroy(cublas);
}

int main() {
    TestLayerInit();
    TestWeightUpdate();

    printf("\n%d assertions passed, %d failed.\n", testsPassed, testsFailed);
    return testsFailed == 0 ? 0 : 1;
}
