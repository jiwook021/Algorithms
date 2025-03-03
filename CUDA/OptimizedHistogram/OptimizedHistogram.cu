/**
 * @file OptimizedHistogram.cu
 * @brief Shared-memory optimized CUDA histogram kernel.
 *
 * Each thread block builds a local histogram in shared memory, then atomically
 * merges into the global histogram. This reduces global memory atomic contention.
 */

#include "OptimizedHistogram.cuh"
#include <cstring>
#include <cstdio>

__global__ void HistogramSharedKernel(const unsigned char* data,
                                     unsigned int* histogram,
                                     int dataSize, int numBins) {
    __shared__ unsigned int sharedHistogram[NUM_BINS];

    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int localTid = threadIdx.x;

    // Initialize shared memory
    for (int i = localTid; i < numBins; i += blockDim.x) {
        sharedHistogram[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop into shared memory
    for (int i = tid; i < dataSize; i += blockDim.x * gridDim.x) {
        if (data[i] < numBins) {
            atomicAdd(&sharedHistogram[data[i]], 1);
        }
    }
    __syncthreads();

    // Merge shared histogram into global
    for (int i = localTid; i < numBins; i += blockDim.x) {
        if (sharedHistogram[i] > 0) {
            atomicAdd(&histogram[i], sharedHistogram[i]);
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
