// image_preprocessing.cu
// CUDA Kernels integrated with OpenCV for processing 'input.png' and saving intermediate outputs

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <opencv2/opencv.hpp>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error: %s (error code: %d)\n",                    \
              cudaGetErrorString(err), err);                                  \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

__global__ void cropKernel(const uint8_t* input, uint8_t* output, int inWidth, int inHeight,
                           int outWidth, int outHeight, int channels,
                           int cropX, int cropY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;

  if (x >= outWidth || y >= outHeight || c >= channels) return;

  int inputIdx = ((y + cropY) * inWidth + (x + cropX)) * channels + c;
  int outputIdx = (y * outWidth + x) * channels + c;

  output[outputIdx] = input[inputIdx];
}

__global__ void resizeKernel(const uint8_t* input, uint8_t* output, int inWidth, int inHeight,
                             int outWidth, int outHeight, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;

  if (x >= outWidth || y >= outHeight || c >= channels) return;

  int srcX = x * inWidth / outWidth;
  int srcY = y * inHeight / outHeight;

  int inputIdx = (srcY * inWidth + srcX) * channels + c;
  int outputIdx = (y * outWidth + x) * channels + c;

  output[outputIdx] = input[inputIdx];
}

__global__ void paddingKernel(const uint8_t* input, uint8_t* output, int inWidth, int inHeight,
                              int outWidth, int outHeight, int channels,
                              int padX, int padY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;

  if (x >= outWidth || y >= outHeight || c >= channels) return;

  int outputIdx = (y * outWidth + x) * channels + c;

  if (x < padX || x >= (padX + inWidth) || y < padY || y >= (padY + inHeight)) {
    output[outputIdx] = 0;
  } else {
    int inputIdx = ((y - padY) * inWidth + (x - padX)) * channels + c;
    output[outputIdx] = input[inputIdx];
  }
}

int main() {
  // Load input image
  cv::Mat inputImg = cv::imread("input.png");
  if (inputImg.empty()) {
    fprintf(stderr, "Could not read input.png\n");
    return EXIT_FAILURE;
  }

  // Image dimensions
  const int inWidth = inputImg.cols;
  const int inHeight = inputImg.rows;
  const int channels = inputImg.channels();

  // Define sizes for each processing step
  const int cropWidth = inWidth / 2, cropHeight = inHeight / 2;
  const int resizeWidth = cropWidth / 2, resizeHeight = cropHeight / 2;
  const int padWidth = resizeWidth + 40, padHeight = resizeHeight + 40;

  // Calculate memory sizes
  size_t inputSize = inWidth * inHeight * channels * sizeof(uint8_t);
  size_t cropSize = cropWidth * cropHeight * channels * sizeof(uint8_t);
  size_t resizeSize = resizeWidth * resizeHeight * channels * sizeof(uint8_t);
  size_t paddedSize = padWidth * padHeight * channels * sizeof(uint8_t);

  // Allocate device memory
  uint8_t *d_input, *d_crop, *d_resize, *d_padded;
  CUDA_CHECK(cudaMalloc(&d_input, inputSize));
  CUDA_CHECK(cudaMalloc(&d_crop, cropSize));
  CUDA_CHECK(cudaMalloc(&d_resize, resizeSize));
  CUDA_CHECK(cudaMalloc(&d_padded, paddedSize));

  // Copy input image to device
  CUDA_CHECK(cudaMemcpy(d_input, inputImg.data, inputSize, cudaMemcpyHostToDevice));

  // Define block and grid dimensions
  dim3 block(16, 16);

  // Crop operation
  dim3 gridCrop((cropWidth + block.x - 1) / block.x, (cropHeight + block.y - 1) / block.y, channels);
  cropKernel<<<gridCrop, block>>>(d_input, d_crop, inWidth, inHeight, cropWidth, cropHeight, channels, inWidth/4, inHeight/4);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Save cropped image
  cv::Mat outputImg1(cropHeight, cropWidth, inputImg.type());
  CUDA_CHECK(cudaMemcpy(outputImg1.data, d_crop, cropSize, cudaMemcpyDeviceToHost));
  cv::imwrite("output1.png", outputImg1);

  // Resize operation
  dim3 gridResize((resizeWidth + block.x - 1) / block.x, (resizeHeight + block.y - 1) / block.y, channels);
  resizeKernel<<<gridResize, block>>>(d_crop, d_resize, cropWidth, cropHeight, resizeWidth, resizeHeight, channels);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Save resized image
  cv::Mat outputImg2(resizeHeight, resizeWidth, inputImg.type());
  CUDA_CHECK(cudaMemcpy(outputImg2.data, d_resize, resizeSize, cudaMemcpyDeviceToHost));
  cv::imwrite("output2.png", outputImg2);

  // Padding operation
  dim3 gridPad((padWidth + block.x - 1) / block.x, (padHeight + block.y - 1) / block.y, channels);
  paddingKernel<<<gridPad, block>>>(d_resize, d_padded, resizeWidth, resizeHeight, padWidth, padHeight, channels, 20, 20);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Save padded image
  cv::Mat outputImg3(padHeight, padWidth, inputImg.type());
  CUDA_CHECK(cudaMemcpy(outputImg3.data, d_padded, paddedSize, cudaMemcpyDeviceToHost));
  cv::imwrite("output3.png", outputImg3);

  // Free device memory
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_crop));
  CUDA_CHECK(cudaFree(d_resize));
  CUDA_CHECK(cudaFree(d_padded));

  printf("Processed and saved output1.png, output2.png, and output3.png\n");
  return 0;
}