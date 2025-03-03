/**
 * @file BatchNormalization.cu
 * @brief CUDA batch normalization kernel implementations.
 *
 * Forward:  output = gamma * (input - mean) / sqrt(var + epsilon) + beta
 * Backward: computes gradients for gamma, beta, and input.
 *
 * Complexity:
 *   - Forward:  O(N) where N = batchSize * featureSize * spatialSize
 *   - Backward: O(N)
 */

#include "BatchNormalization.cuh"
#include <cmath>

__global__ void ComputeMeanKernel(const float* input, float* mean,
                                  int batchSize, int featureSize, int spatialSize) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (featureIdx < featureSize) {
        float sum = 0.0f;
        int count = batchSize * spatialSize;
        for (int n = 0; n < batchSize; ++n) {
            for (int s = 0; s < spatialSize; ++s) {
                int idx = n * featureSize * spatialSize + featureIdx * spatialSize + s;
                sum += input[idx];
            }
        }
        mean[featureIdx] = sum / count;
    }
}

__global__ void ComputeVarianceKernel(const float* input, const float* mean, float* var,
                                      int batchSize, int featureSize, int spatialSize) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (featureIdx < featureSize) {
        float sumSqDiff = 0.0f;
        int count = batchSize * spatialSize;
        float featureMean = mean[featureIdx];
        for (int n = 0; n < batchSize; ++n) {
            for (int s = 0; s < spatialSize; ++s) {
                int idx = n * featureSize * spatialSize + featureIdx * spatialSize + s;
                float diff = input[idx] - featureMean;
                sumSqDiff += diff * diff;
            }
        }
        var[featureIdx] = sumSqDiff / count;
    }
}

__global__ void UpdateRunningStatsKernel(const float* batchMean, const float* batchVar,
                                         float* runningMean, float* runningVar,
                                         float momentum, int featureSize) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (featureIdx < featureSize) {
        runningMean[featureIdx] = momentum * runningMean[featureIdx] +
                                  (1.0f - momentum) * batchMean[featureIdx];
        runningVar[featureIdx]  = momentum * runningVar[featureIdx] +
                                  (1.0f - momentum) * batchVar[featureIdx];
    }
}

__global__ void BatchNormForwardKernel(const float* input, float* output,
                                       const float* mean, const float* var,
                                       const float* gamma, const float* beta,
                                       float epsilon, int batchSize, int featureSize, int spatialSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchSize * featureSize * spatialSize;
    if (idx < totalElements) {
        int f = (idx % (featureSize * spatialSize)) / spatialSize;
        float normalized = (input[idx] - mean[f]) / sqrtf(var[f] + epsilon);
        output[idx] = gamma[f] * normalized + beta[f];
    }
}

__global__ void ComputeParamGradientsKernel(const float* input, const float* gradOutput,
                                            const float* mean, const float* var,
                                            float* gradGamma, float* gradBeta,
                                            float epsilon, int batchSize, int featureSize, int spatialSize) {
    int featureIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (featureIdx < featureSize) {
        float sumDy = 0.0f, sumDyXhat = 0.0f;
        float featureMean = mean[featureIdx];
        float invStd = 1.0f / sqrtf(var[featureIdx] + epsilon);
        for (int n = 0; n < batchSize; ++n) {
            for (int s = 0; s < spatialSize; ++s) {
                int idx = n * featureSize * spatialSize + featureIdx * spatialSize + s;
                float xhat = (input[idx] - featureMean) * invStd;
                sumDy += gradOutput[idx];
                sumDyXhat += gradOutput[idx] * xhat;
            }
        }
        gradBeta[featureIdx] = sumDy;
        gradGamma[featureIdx] = sumDyXhat;
    }
}

__global__ void ComputeInputGradientKernel(const float* input, const float* gradOutput,
                                           const float* mean, const float* var,
                                           const float* gamma, float* gradInput,
                                           float epsilon, int batchSize, int featureSize, int spatialSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchSize * featureSize * spatialSize;
    if (idx < totalElements) {
        int f = (idx % (featureSize * spatialSize)) / spatialSize;
        float featureMean = mean[f];
        float featureGamma = gamma[f];
        float invStd = 1.0f / sqrtf(var[f] + epsilon);
        int count = batchSize * spatialSize;
        float xCentered = input[idx] - featureMean;
        float dy = gradOutput[idx];

        float sumDy = 0.0f, sumDyXhat = 0.0f;
        for (int n = 0; n < batchSize; ++n) {
            for (int s = 0; s < spatialSize; ++s) {
                int innerIdx = n * featureSize * spatialSize + f * spatialSize + s;
                sumDy += gradOutput[innerIdx];
                sumDyXhat += gradOutput[innerIdx] * (input[innerIdx] - featureMean) * invStd;
            }
        }
        gradInput[idx] = featureGamma * invStd * (dy - sumDy / count - xCentered * invStd * invStd * sumDyXhat / count);
    }
}
