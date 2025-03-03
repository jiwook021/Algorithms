#ifndef MAX_POOLING_CUH
#define MAX_POOLING_CUH

#include <cuda_runtime.h>
#include <vector>

/**
 * @brief Error-checking macro for CUDA API calls.
 */
#define CHECK_CUDA(Call)                                                      \
    do {                                                                      \
        cudaError_t Err = Call;                                               \
        if (Err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(Err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/**
 * @brief CUDA kernel that computes 2D max-pooling for a single-channel feature map.
 *
 * Each thread computes one output element by scanning a poolWidth x poolHeight
 * window in the input and writing the maximum value to the output.
 */
__global__ void MaxPoolingKernel(const float* Input, float* Output,
                                 int InputWidth, int InputHeight,
                                 int PoolWidth, int PoolHeight,
                                 int OutputWidth, int OutputHeight);

/**
 * @brief Host API: performs 2D max-pooling on a single-channel feature map using CUDA.
 *
 * Output dimensions are computed with ceiling division so that edge elements
 * are included even when input dimensions are not divisible by pool size.
 */
void MaxPooling(const std::vector<float>& Input, std::vector<float>& Output,
                int InputWidth, int InputHeight,
                int PoolWidth, int PoolHeight);

#endif // MAX_POOLING_CUH
