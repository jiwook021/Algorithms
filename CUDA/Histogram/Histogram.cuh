#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

/**
 * @file Histogram.cuh
 * @brief Header for CUDA histogram computation kernel.
 *
 * Computes histograms using atomic operations on the GPU with a
 * grid-stride loop pattern. Includes CPU reference implementation for verification.
 */

#include <cuda_runtime.h>

#define CUDA_CHECK(Call)                                                      \
    do {                                                                      \
        cudaError_t Err = Call;                                               \
        if (Err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(Err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define NUM_BINS 256
#define MAX_PIXEL_VALUE 255
#define BLOCK_SIZE 256

/**
 * @brief CUDA kernel for histogram calculation using atomicAdd.
 *
 * @param data      Input data array (unsigned char values).
 * @param histogram Output histogram array (NUM_BINS bins).
 * @param dataSize  Number of elements in the input data.
 * @param numBins   Number of histogram bins.
 */
__global__ void HistogramKernel(const unsigned char* data, unsigned int* histogram,
                                int dataSize, int numBins);

/**
 * @brief CPU reference implementation of histogram for verification.
 */
void ComputeHistogramCpu(const unsigned char* data, unsigned int* histogram,
                         int dataSize, int numBins);

/**
 * @brief Compare CPU and GPU histograms for verification.
 * @return true if histograms match, false otherwise.
 */
bool CompareHistograms(const unsigned int* cpuHist, const unsigned int* gpuHist,
                       int numBins);

#endif // HISTOGRAM_CUH
