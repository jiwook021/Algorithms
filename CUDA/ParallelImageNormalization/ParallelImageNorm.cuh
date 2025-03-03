#ifndef PARALLEL_IMAGE_NORM_CUH
#define PARALLEL_IMAGE_NORM_CUH

/**
 * @file ParallelImageNorm.cuh
 * @brief Header for CUDA parallel image normalization kernels.
 *
 * Pipeline: uchar->float -> sum reduction -> mean -> variance -> normalize
 */

#include <cuda_runtime.h>

/**
 * @brief Convert unsigned char image to float.
 */
__global__ void ConvertUcharToFloat(unsigned char* dImageUchar, float* dImageFloat, int size);

/**
 * @brief Parallel sum reduction per channel for computing mean.
 */
__global__ void SumReductionKernel(float* dImageFloat, float* dPartialSums, int n, int c);

/**
 * @brief Parallel sum-of-squares reduction per channel for computing variance.
 */
__global__ void SumOfSquaresReductionKernel(float* dImageFloat, float* dMean,
                                            float* dPartialSums, int n, int c);

/**
 * @brief Per-pixel z-score normalization: (pixel - mean) / std.
 */
__global__ void NormalizeKernel(float* dImageFloat, float* dMean, float* dStddev,
                                float* dNormalized, int size, int c);

#endif // PARALLEL_IMAGE_NORM_CUH
