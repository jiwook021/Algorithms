/**
 * @file Histogram.cu
 * @brief CUDA histogram computation kernel and CPU reference implementation.
 *
 * Uses atomicAdd for race-free bin counting with a grid-stride loop pattern.
 *
 * Complexity:
 *   - Time:  O(N / (blocks * threads)) per thread
 *   - Space: O(NUM_BINS) for the histogram on device
 */

#include "Histogram.cuh"
#include <cstdio>
#include <cstring>

__global__ void HistogramKernel(const unsigned char* data, unsigned int* histogram,
                                int dataSize, int numBins) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < dataSize; i += stride) {
        if (data[i] < numBins) {
            atomicAdd(&histogram[data[i]], 1);
        }
    }
}

void ComputeHistogramCpu(const unsigned char* data, unsigned int* histogram,
                         int dataSize, int numBins) {
    memset(histogram, 0, numBins * sizeof(unsigned int));
    for (int i = 0; i < dataSize; i++) {
        if (data[i] < numBins) {
            histogram[data[i]]++;
        }
    }
}

bool CompareHistograms(const unsigned int* cpuHist, const unsigned int* gpuHist,
                       int numBins) {
    for (int i = 0; i < numBins; i++) {
        if (cpuHist[i] != gpuHist[i]) {
            printf("Mismatch at bin %d: CPU = %u, GPU = %u\n", i, cpuHist[i], gpuHist[i]);
            return false;
        }
    }
    return true;
}
