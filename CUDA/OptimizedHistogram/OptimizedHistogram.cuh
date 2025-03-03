#ifndef OPTIMIZED_HISTOGRAM_CUH
#define OPTIMIZED_HISTOGRAM_CUH

/**
 * @file OptimizedHistogram.cuh
 * @brief Header for CUDA optimized histogram using shared memory.
 *
 * Each block maintains a local histogram in shared memory, then atomically
 * merges into the global histogram -- reducing global memory contention.
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
 * @brief Shared-memory optimized histogram kernel.
 */
__global__ void HistogramSharedKernel(const unsigned char* data,
                                     unsigned int* histogram,
                                     int dataSize, int numBins);

/**
 * @brief CPU reference histogram for verification.
 */
void ComputeHistogramCpu(const unsigned char* data, unsigned int* histogram,
                         int dataSize, int numBins);

/**
 * @brief Compare two histograms.
 */
bool CompareHistograms(const unsigned int* cpuHist, const unsigned int* gpuHist,
                       int numBins);

#endif // OPTIMIZED_HISTOGRAM_CUH
