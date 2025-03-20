# Step-by-Step Explanation: main.cu

Let’s break down the code **step by step**, explaining every significant section in detail. I’ll use simple language, examples, and diagrams to make everything clear, even for beginners.

---

### **1. Header Files and Macros**
```cuda
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    if((call) != cudaSuccess) {                                             \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__;       \
        std::cerr << " code=" << cudaGetErrorString(call) << std::endl;     \
        exit(EXIT_FAILURE);                                                 \
    }
```

#### **What It Does**
- **Header Files**:
  - `cuda_runtime.h`: Provides CUDA-specific functions and types for GPU programming.
  - `iostream`: Used for printing messages to the console (e.g., errors or progress updates).
  - `vector`: A C++ container for storing dynamic arrays (used for the kernel).
  - `opencv2/opencv.hpp`: OpenCV library for image processing (loading, saving, and manipulating images).

- **Macro**:
  - `CUDA_CHECK`: A helper macro to check for errors in CUDA function calls. If a CUDA function fails, it prints an error message and exits the program.

#### **Why It’s Used**
- **Error Checking**: CUDA functions can fail for many reasons (e.g., out of memory, invalid arguments). This macro ensures that errors are caught immediately, making debugging easier.
- **Convenience**: Instead of writing error-checking code for every CUDA call, the macro simplifies the process.

---

### **2. CUDA Kernel: `conv2D`**
```cuda
__global__ void conv2D(const float* input, const float* kernel, float* output,
                       int inputWidth, int inputHeight,
                       int kernelWidth, int kernelHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    if (x >= outputWidth || y >= outputHeight) return;

    float sum = 0.0f;
    for (int ky = 0; ky < kernelHeight; ky++) {
        for (int kx = 0; kx < kernelWidth; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            sum += input[iy * inputWidth + ix] * kernel[ky * kernelWidth + kx];
        }
    }
    output[y * outputWidth + x] = sum;
}
```

#### **What It Does**
- This is the **CUDA kernel**, a function that runs on the GPU. It performs 2D convolution on the input image using the provided kernel.

#### **Step-by-Step Breakdown**
1. **Thread and Block Indexing**:
   - `x` and `y` are calculated to determine which pixel in the output image this thread is responsible for.
   - `blockIdx.x` and `blockIdx.y`: The block’s position in the grid.
   - `blockDim.x` and `blockDim.y`: The number of threads in a block.
   - `threadIdx.x` and `threadIdx.y`: The thread’s position within its block.

   Example:
   - If `blockIdx.x = 1`, `blockDim.x = 16`, and `threadIdx.x = 2`, then `x = 1 * 16 + 2 = 18`.

2. **Output Dimensions**:
   - The output image is smaller than the input image because the kernel cannot slide past the edges.
   - `outputWidth = inputWidth - kernelWidth + 1`
   - `outputHeight = inputHeight - kernelHeight + 1`

3. **Boundary Check**:
   - If the thread’s `x` or `y` is outside the output image dimensions, the thread exits early (`return`).

4. **Convolution Calculation**:
   - Each thread computes the convolution result for one pixel in the output image.
   - Two nested loops iterate over the kernel dimensions:
     - `ky` and `kx` are the kernel’s row and column indices.
     - `ix` and `iy` are the corresponding positions in the input image.
     - The weighted sum is computed: `sum += input[iy * inputWidth + ix] * kernel[ky * kernelWidth + kx]`.

5. **Store Result**:
   - The computed sum is stored in the output image at position `(y, x)`.

#### **Why It’s Used**
- **Parallelism**: Each thread computes one pixel independently, allowing the GPU to process thousands of pixels simultaneously.
- **Efficiency**: The nested loops are optimized for small kernels (e.g., 3x3), making the computation fast.

---

### **3. Helper Function: `runConvolution`**
```cuda
void runConvolution(const cv::Mat& inputImg, const std::vector<float>& kernel, cv::Mat& outputImg,
                    int kernelWidth, int kernelHeight)
{
    int inputWidth = inputImg.cols;
    int inputHeight = inputImg.rows;
    int outputWidth = inputWidth - kernelWidth + 1;
    int outputHeight = inputHeight - kernelHeight + 1;

    size_t inputSize = inputWidth * inputHeight * sizeof(float);
    size_t kernelSize = kernelWidth * kernelHeight * sizeof(float);
    size_t outputSize = outputWidth * outputHeight * sizeof(float);

    // Allocate GPU memory
    float *d_input, *d_kernel, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, inputSize));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelSize));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_input, inputImg.ptr<float>(), inputSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernelSize, cudaMemcpyHostToDevice));

    // Define grid and block size
    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x,
                  (outputHeight + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    conv2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,
                                    inputWidth, inputHeight, kernelWidth, kernelHeight);
    CUDA_CHECK(cudaGetLastError());

    // Copy back result
    outputImg.create(outputHeight, outputWidth, CV_32FC1);
    CUDA_CHECK(cudaMemcpy(outputImg.ptr<float>(), d_output, outputSize, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));
}
```

#### **What It Does**
- This function manages the GPU-side operations for convolution:
  - Allocates memory on the GPU.
  - Copies data to the GPU.
  - Launches the CUDA kernel.
  - Copies the result back to the CPU.
  - Frees GPU memory.

#### **Step-by-Step Breakdown**
1. **Input and Output Dimensions**:
   - `inputWidth` and `inputHeight` are the dimensions of the input image.
   - `outputWidth` and `outputHeight` are calculated based on the kernel size.

2. **Memory Allocation**:
   - `cudaMalloc` allocates memory on the GPU for the input image, kernel, and output image.

3. **Data Transfer**:
   - `cudaMemcpy` copies the input image and kernel from the CPU to the GPU.

4. **Grid and Block Dimensions**:
   - `blockSize` is set to 16x16 threads per block.
   - `gridSize` is calculated to cover the entire output image.

5. **Kernel Launch**:
   - The `conv2D` kernel is launched with the specified grid and block dimensions.

6. **Result Transfer**:
   - The output image is copied back from the GPU to the CPU.

7. **Memory Cleanup**:
   - `cudaFree` releases the allocated GPU memory.

#### **Why It’s Used**
- **Memory Management**: Ensures efficient use of GPU memory.
- **Data Transfer**: Moves data between the CPU and GPU as needed.
- **Kernel Launch**: Executes the convolution operation on the GPU.

---

### **4. Main Function**
```cuda
int main()
{
    // Load input image as grayscale and convert to float
    cv::Mat inputImg = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    if (inputImg.empty()) {
        std::cerr << "Failed to load input.png!" << std::endl;
        return -1;
    }

    inputImg.convertTo(inputImg, CV_32FC1, 1.0 / 255.0);

    // Define a simple Sobel kernel (horizontal edge detection)
    std::vector<float> kernel = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    int kernelWidth = 3;
    int kernelHeight = 3;

    // Output image
    cv::Mat outputImg;

    // Run convolution
    runConvolution(inputImg, kernel, outputImg, kernelWidth, kernelHeight);

    // Normalize and convert back to 8-bit image
    cv::normalize(outputImg, outputImg, 0, 255, cv::NORM_MINMAX);
    outputImg.convertTo(outputImg, CV_8UC1);

    // Save the output image
    if (!cv::imwrite("output.png", outputImg)) {
        std::cerr << "Failed to save output.png!" << std::endl;
        return -1;
    }

    std::cout << "Convolution completed. Check output.png" << std::endl;
    return 0;
}
```

#### **What It Does**
- The `main` function orchestrates the entire program:
  - Loads the input image.
  - Defines the Sobel kernel.
  - Calls `runConvolution` to perform the convolution.
  - Postprocesses and saves the output image.

#### **Step-by-Step Breakdown**
1. **Load Input Image**:
   - `cv::imread` loads the image as grayscale.
   - `convertTo` normalizes the pixel values to the range [0, 1].

2. **Define Kernel**:
   - The Sobel kernel is defined as a 3x3 matrix for horizontal edge detection.

3. **Run Convolution**:
   - `runConvolution` is called to perform the convolution on the GPU.

4. **Postprocess Output**:
   - `cv::normalize` scales the output to the range [0, 255].
   - `convertTo` converts the image back to 8-bit format.

5. **Save Output**:
   - `cv::imwrite` saves the output image as `output.png`.

#### **Why It’s Used**
- **Image Processing**: Prepares the image for convolution and saves the result.
- **Edge Detection**: The Sobel kernel highlights edges in the image.

---

### **Summary**
This code demonstrates how to perform 2D convolution on an image using CUDA. It loads an image, applies a Sobel kernel for edge detection, and saves the result. The GPU’s parallelism is leveraged to accelerate the computation, making it much faster than a CPU implementation. Each part of the code is carefully designed to handle memory management, error checking, and data transfer, ensuring robust and efficient execution.