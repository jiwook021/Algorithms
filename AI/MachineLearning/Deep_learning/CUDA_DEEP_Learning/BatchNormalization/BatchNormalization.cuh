#ifndef BATCH_NORMALIZATION_CUH
#define BATCH_NORMALIZATION_CUH

/**
 * @file BatchNormalization.cuh
 * @brief Header for CUDA batch normalization layer.
 *
 * Provides forward/backward pass kernels for batch normalization
 * with learnable gamma/beta parameters.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(Call)                                                      \
    do {                                                                      \
        cudaError_t Err = Call;                                               \
        if (Err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(Err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

constexpr float BN_EPSILON = 1e-5f;

__global__ void ComputeMeanKernel(const float* input, float* mean,
                                  int batchSize, int featureSize, int spatialSize);

__global__ void ComputeVarianceKernel(const float* input, const float* mean, float* var,
                                      int batchSize, int featureSize, int spatialSize);

__global__ void UpdateRunningStatsKernel(const float* batchMean, const float* batchVar,
                                         float* runningMean, float* runningVar,
                                         float momentum, int featureSize);

__global__ void BatchNormForwardKernel(const float* input, float* output,
                                       const float* mean, const float* var,
                                       const float* gamma, const float* beta,
                                       float epsilon, int batchSize, int featureSize, int spatialSize);

__global__ void ComputeParamGradientsKernel(const float* input, const float* gradOutput,
                                            const float* mean, const float* var,
                                            float* gradGamma, float* gradBeta,
                                            float epsilon, int batchSize, int featureSize, int spatialSize);

__global__ void ComputeInputGradientKernel(const float* input, const float* gradOutput,
                                           const float* mean, const float* var,
                                           const float* gamma, float* gradInput,
                                           float epsilon, int batchSize, int featureSize, int spatialSize);

#endif // BATCH_NORMALIZATION_CUH
