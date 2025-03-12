#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include <mutex>

// Define STB_IMAGE_IMPLEMENTATION before including to create the implementation
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Define STB_IMAGE_WRITE_IMPLEMENTATION before including to create the implementation
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Error handling macro for CUDA calls
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Predefined 3x3 filters
const float GAUSSIAN_BLUR_3x3[9] = {
    1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
    2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
    1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f
};

const float EDGE_DETECTION_3x3[9] = {
    -1.0f, -1.0f, -1.0f,
    -1.0f,  8.0f, -1.0f,
    -1.0f, -1.0f, -1.0f
};

const float SOBEL_X_3x3[9] = {
    -1.0f, 0.0f, 1.0f,
    -2.0f, 0.0f, 2.0f,
    -1.0f, 0.0f, 1.0f
};

const float SOBEL_Y_3x3[9] = {
    -1.0f, -2.0f, -1.0f,
     0.0f,  0.0f,  0.0f,
     1.0f,  2.0f,  1.0f
};

// Static mutex for thread-safe error reporting
// This is needed to prevent race conditions when multiple threads call CUDA functions
static std::mutex cuda_mutex;

/**
 * CUDA kernel for applying convolution to an image
 * 
 * @param inputImage   Input image data in global memory
 * @param outputImage  Output image data in global memory
 * @param filter       Convolution filter/kernel in constant memory
 * @param filterWidth  Width of the filter (assuming square filter)
 * @param width        Width of the image
 * @param height       Height of the image
 * @param channels     Number of channels in the image
 */
__global__ void convolutionKernel(
    const unsigned char* inputImage,
    unsigned char* outputImage,
    const float* filter,
    int filterWidth,
    int width,
    int height,
    int channels
) {
    // Calculate pixel position based on thread and block indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if we're within the image bounds
    if (x < width && y < height) {
        // Process each channel separately
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            // Apply convolution filter to the neighborhood
            int filterRadius = filterWidth / 2;
            
            for (int fy = -filterRadius; fy <= filterRadius; fy++) {
                for (int fx = -filterRadius; fx <= filterRadius; fx++) {
                    // Calculate neighbor coordinates with boundary checking
                    int nx = min(max(x + fx, 0), width - 1);
                    int ny = min(max(y + fy, 0), height - 1);
                    
                    // Get value from input image
                    unsigned char pixel = inputImage[(ny * width + nx) * channels + c];
                    
                    // Get corresponding filter value
                    float filterValue = filter[(fy + filterRadius) * filterWidth + (fx + filterRadius)];
                    
                    // Accumulate weighted sum
                    sum += pixel * filterValue;
                }
            }
            
            // Clamp the result to valid pixel range [0, 255]
            int result = min(max(int(sum), 0), 255);
            
            // Write to output image
            outputImage[(y * width + x) * channels + c] = (unsigned char)result;
        }
    }
}

/**
 * Fixed CUDA kernel for applying convolution using shared memory
 * 
 * @param inputImage   Input image data in global memory
 * @param outputImage  Output image data in global memory
 * @param filter       Convolution filter/kernel in constant memory
 * @param filterWidth  Width of the filter (assuming square filter)
 * @param width        Width of the image
 * @param height       Height of the image
 * @param channels     Number of channels in the image
 */
__global__ void convolutionSharedKernel(
    const unsigned char* inputImage,
    unsigned char* outputImage,
    const float* filter,
    int filterWidth,
    int width,
    int height,
    int channels
) {
    // Define shared memory for the image tile
    extern __shared__ unsigned char sharedMem[];
    
    // Calculate pixel position based on thread and block indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int filterRadius = filterWidth / 2;
    
    // Define dimensions for the shared memory tile
    int tileDimX = blockDim.x + 2 * filterRadius;
    int tileDimY = blockDim.y + 2 * filterRadius;
    
    // Load image data into shared memory, including halo regions
    for (int c = 0; c < channels; c++) {
        // Each thread loads its own center pixel
        if (x < width && y < height) {
            sharedMem[((ty + filterRadius) * tileDimX + (tx + filterRadius)) * channels + c] = 
                inputImage[(y * width + x) * channels + c];
        }
        else {
            // Out of bounds, set to zero or clamp to edge
            sharedMem[((ty + filterRadius) * tileDimX + (tx + filterRadius)) * channels + c] = 0;
        }

        // Load halo region pixels (top and bottom rows)
        if (ty < filterRadius) {
            // Top halo region
            int srcY = y - filterRadius;
            if (srcY >= 0 && x < width) {
                sharedMem[(ty * tileDimX + (tx + filterRadius)) * channels + c] = 
                    inputImage[(srcY * width + x) * channels + c];
            }
            else {
                sharedMem[(ty * tileDimX + (tx + filterRadius)) * channels + c] = 0;
            }
            
            // Bottom halo region (using threads at the top of the block)
            srcY = y + blockDim.y;
            if (srcY < height && x < width) {
                sharedMem[((ty + blockDim.y + filterRadius) * tileDimX + (tx + filterRadius)) * channels + c] = 
                    inputImage[(srcY * width + x) * channels + c];
            }
            else {
                sharedMem[((ty + blockDim.y + filterRadius) * tileDimX + (tx + filterRadius)) * channels + c] = 0;
            }
        }
        
        // Load halo region pixels (left and right columns)
        if (tx < filterRadius) {
            // Left halo region
            int srcX = x - filterRadius;
            if (srcX >= 0 && y < height) {
                sharedMem[((ty + filterRadius) * tileDimX + tx) * channels + c] = 
                    inputImage[(y * width + srcX) * channels + c];
            }
            else {
                sharedMem[((ty + filterRadius) * tileDimX + tx) * channels + c] = 0;
            }
            
            // Right halo region (using threads at the left of the block)
            srcX = x + blockDim.x;
            if (srcX < width && y < height) {
                sharedMem[((ty + filterRadius) * tileDimX + (tx + blockDim.x + filterRadius)) * channels + c] = 
                    inputImage[(y * width + srcX) * channels + c];
            }
            else {
                sharedMem[((ty + filterRadius) * tileDimX + (tx + blockDim.x + filterRadius)) * channels + c] = 0;
            }
        }
        
        // Load corner halo regions (using threads at the corners of the block)
        if (tx < filterRadius && ty < filterRadius) {
            // Top-left corner
            int srcX = x - filterRadius;
            int srcY = y - filterRadius;
            if (srcX >= 0 && srcY >= 0) {
                sharedMem[(ty * tileDimX + tx) * channels + c] = 
                    inputImage[(srcY * width + srcX) * channels + c];
            }
            else {
                sharedMem[(ty * tileDimX + tx) * channels + c] = 0;
            }
            
            // Top-right corner
            srcX = x + blockDim.x;
            srcY = y - filterRadius;
            if (srcX < width && srcY >= 0) {
                sharedMem[(ty * tileDimX + (tx + blockDim.x + filterRadius)) * channels + c] = 
                    inputImage[(srcY * width + srcX) * channels + c];
            }
            else {
                sharedMem[(ty * tileDimX + (tx + blockDim.x + filterRadius)) * channels + c] = 0;
            }
            
            // Bottom-left corner
            srcX = x - filterRadius;
            srcY = y + blockDim.y;
            if (srcX >= 0 && srcY < height) {
                sharedMem[((ty + blockDim.y + filterRadius) * tileDimX + tx) * channels + c] = 
                    inputImage[(srcY * width + srcX) * channels + c];
            }
            else {
                sharedMem[((ty + blockDim.y + filterRadius) * tileDimX + tx) * channels + c] = 0;
            }
            
            // Bottom-right corner
            srcX = x + blockDim.x;
            srcY = y + blockDim.y;
            if (srcX < width && srcY < height) {
                sharedMem[((ty + blockDim.y + filterRadius) * tileDimX + (tx + blockDim.x + filterRadius)) * channels + c] = 
                    inputImage[(srcY * width + srcX) * channels + c];
            }
            else {
                sharedMem[((ty + blockDim.y + filterRadius) * tileDimX + (tx + blockDim.x + filterRadius)) * channels + c] = 0;
            }
        }
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Process only within image bounds
    if (x < width && y < height) {
        // Process each channel separately
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            // Apply convolution filter to the neighborhood in shared memory
            for (int fy = -filterRadius; fy <= filterRadius; fy++) {
                for (int fx = -filterRadius; fx <= filterRadius; fx++) {
                    // Calculate shared memory index
                    int sharedX = tx + fx + filterRadius;
                    int sharedY = ty + fy + filterRadius;
                    int sharedIdx = (sharedY * tileDimX + sharedX) * channels + c;
                    
                    // Get corresponding filter value
                    float filterValue = filter[(fy + filterRadius) * filterWidth + (fx + filterRadius)];
                    
                    // Accumulate weighted sum
                    sum += sharedMem[sharedIdx] * filterValue;
                }
            }
            
            // Clamp the result to valid pixel range [0, 255]
            int result = min(max(int(sum), 0), 255);
            
            // Write to output image
            outputImage[(y * width + x) * channels + c] = (unsigned char)result;
        }
    }
}

/**
 * Host function to apply convolution to an image
 * 
 * @param h_inputImage   Input image on host
 * @param h_outputImage  Output image on host
 * @param h_filter       Convolution filter/kernel on host
 * @param filterWidth    Width of the filter (assuming square filter)
 * @param width          Width of the image
 * @param height         Height of the image
 * @param channels       Number of channels in the image
 * 
 * Time Complexity: O(width * height * filterWidth^2) operations but parallelized across GPU threads
 * Space Complexity: O(width * height * channels) for input and output images
 */
void applyConvolution(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImage,
    const float* h_filter,
    int filterWidth,
    int width,
    int height,
    int channels
) {
    // Lock for thread-safety during CUDA operations
    std::lock_guard<std::mutex> lock(cuda_mutex);
    
    // Calculate image size in bytes
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    
    // Allocate device memory
    unsigned char* d_inputImage = nullptr;
    unsigned char* d_outputImage = nullptr;
    float* d_filter = nullptr;
    
    CUDA_CHECK(cudaMalloc((void**)&d_inputImage, imageSize));
    CUDA_CHECK(cudaMalloc((void**)&d_outputImage, imageSize));
    CUDA_CHECK(cudaMalloc((void**)&d_filter, filterSize));
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    // For larger images, optimize these values based on GPU architecture
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Launch the convolution kernel
    convolutionKernel<<<gridSize, blockSize>>>(
        d_inputImage,
        d_outputImage,
        d_filter,
        filterWidth,
        width,
        height,
        channels
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_inputImage));
    CUDA_CHECK(cudaFree(d_outputImage));
    CUDA_CHECK(cudaFree(d_filter));
}

/**
 * Host function to apply optimized convolution to an image using shared memory
 * 
 * @param h_inputImage   Input image on host
 * @param h_outputImage  Output image on host
 * @param h_filter       Convolution filter/kernel on host
 * @param filterWidth    Width of the filter (assuming square filter)
 * @param width          Width of the image
 * @param height         Height of the image
 * @param channels       Number of channels in the image
 * 
 * Time Complexity: O(width * height * filterWidth^2) operations but parallelized across GPU threads
 * Space Complexity: O(width * height * channels) for input and output images, plus shared memory per block
 */
void applyConvolutionOptimized(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImage,
    const float* h_filter,
    int filterWidth,
    int width,
    int height,
    int channels
) {
    // Lock for thread-safety during CUDA operations
    std::lock_guard<std::mutex> lock(cuda_mutex);
    
    // Calculate image size in bytes
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    
    // Allocate device memory
    unsigned char* d_inputImage = nullptr;
    unsigned char* d_outputImage = nullptr;
    float* d_filter = nullptr;
    
    CUDA_CHECK(cudaMalloc((void**)&d_inputImage, imageSize));
    CUDA_CHECK(cudaMalloc((void**)&d_outputImage, imageSize));
    CUDA_CHECK(cudaMalloc((void**)&d_filter, filterSize));
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Calculate the filter radius
    int filterRadius = filterWidth / 2;
    
    // Calculate shared memory size
    int tileDimX = blockSize.x + 2 * filterRadius;
    int tileDimY = blockSize.y + 2 * filterRadius;
    size_t sharedMemSize = tileDimX * tileDimY * channels * sizeof(unsigned char);
    
    // Launch the optimized convolution kernel with shared memory
    convolutionSharedKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_inputImage,
        d_outputImage,
        d_filter,
        filterWidth,
        width,
        height,
        channels
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_inputImage));
    CUDA_CHECK(cudaFree(d_outputImage));
    CUDA_CHECK(cudaFree(d_filter));
}

/**
 * Creates a test image (gradient pattern)
 * 
 * @param image     Pointer to image data buffer
 * @param width     Width of the image
 * @param height    Height of the image
 * @param channels  Number of channels in the image
 */
void createTestImage(unsigned char* image, int width, int height, int channels) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                // Create a simple gradient pattern
                image[(y * width + x) * channels + c] = (unsigned char)((x + y) % 256);
            }
        }
    }
}

/**
 * Load an image from a file using stb_image library
 * 
 * @param filename  Name of the image file to load
 * @param width     Pointer to store the width of the loaded image
 * @param height    Pointer to store the height of the loaded image
 * @param channels  Pointer to store the number of channels in the loaded image
 * @return          Pointer to the loaded image data, or nullptr on failure
 */
unsigned char* loadImage(const char* filename, int* width, int* height, int* channels) {
    // Load the image
    unsigned char* image = stbi_load(filename, width, height, channels, 0);
    
    if (!image) {
        std::cerr << "Error: Could not load image file: " << filename << std::endl;
        std::cerr << "stbi error: " << stbi_failure_reason() << std::endl;
        return nullptr;
    }
    
    std::cout << "Loaded image: " << filename << " (" << *width << "x" << *height 
              << ", " << *channels << " channels)" << std::endl;
    
    return image;
}

/**
 * Save an image to a JPG file using stb_image_write library
 * 
 * @param filename  Name of the output file
 * @param image     Pointer to image data
 * @param width     Width of the image
 * @param height    Height of the image
 * @param channels  Number of channels in the image
 * @param quality   JPEG quality (1-100)
 * @return          True on success, false on failure
 */
bool saveJpg(const char* filename, const unsigned char* image, int width, int height, int channels, int quality = 90) {
    if (!image) {
        std::cerr << "Error: Null image data" << std::endl;
        return false;
    }
    
    int result = stbi_write_jpg(filename, width, height, channels, image, quality);
    
    if (!result) {
        std::cerr << "Error: Could not save JPG file: " << filename << std::endl;
        return false;
    }
    
    std::cout << "Saved image: " << filename << std::endl;
    return true;
}

/**
 * Benchmarks the performance of a convolution function
 * 
 * @param convFunc    Function pointer to the convolution function
 * @param inputImage  Input image data
 * @param outputImage Output image data
 * @param filter      Convolution filter
 * @param filterWidth Width of the filter
 * @param width       Width of the image
 * @param height      Height of the image
 * @param channels    Number of channels in the image
 * @param iterations  Number of iterations for benchmarking
 * @return            Average execution time in seconds
 */
template<typename ConvFunc>
double benchmarkConvolution(
    ConvFunc convFunc,
    const unsigned char* inputImage,
    unsigned char* outputImage,
    const float* filter,
    int filterWidth,
    int width,
    int height,
    int channels,
    int iterations = 10
) {
    // Warm-up run
    convFunc(inputImage, outputImage, filter, filterWidth, width, height, channels);
    
    // Benchmark runs
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        convFunc(inputImage, outputImage, filter, filterWidth, width, height, channels);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    return elapsed.count() / iterations;  // Average time per iteration
}

/**
 * Displays usage information for the program
 * 
 * @param programName Name of the program executable
 */
void displayUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [input_image.jpg]" << std::endl;
    std::cout << "If no input image is provided, a test image will be generated." << std::endl;
    std::cout << "The program will apply various convolution filters and save the results as JPG files." << std::endl;
}

int main(int argc, char** argv) {
    // Check if we're processing an input file or generating a test image
    bool useTestImage = (argc <= 1);
    const char* inputFilename = (argc > 1) ? argv[1] : nullptr;
    
    // Define filter parameters
    const int filterWidth = 3;  // 3x3 filter
    
    // Variables for image dimensions
    int width, height, channels;
    unsigned char* inputImage = nullptr;
    
    // Allocate memory for input and output images
    if (useTestImage) {
        // Use a test image
        width = 1024;
        height = 1024;
        channels = 3;  // RGB
        
        size_t imageSize = width * height * channels;
        inputImage = new unsigned char[imageSize];
        
        // Create a test image
        std::cout << "Creating test image..." << std::endl;
        createTestImage(inputImage, width, height, channels);
        
        // Save the original test image
        saveJpg("original.jpg", inputImage, width, height, channels);
    } else {
        // Load the input image
        inputImage = loadImage(inputFilename, &width, &height, &channels);
        
        if (!inputImage) {
            std::cerr << "Failed to load input image: " << inputFilename << std::endl;
            return 1;
        }
    }
    
    // Allocate memory for output images
    size_t imageSize = width * height * channels;
    std::vector<unsigned char> outputImageBasic(imageSize);
    std::vector<unsigned char> outputImageOptimized(imageSize);
    std::vector<unsigned char> outputImageEdge(imageSize);
    std::vector<unsigned char> outputImageSobelX(imageSize);
    std::vector<unsigned char> outputImageSobelY(imageSize);
    
    try {
        std::cout << "Benchmarking convolution operations..." << std::endl;
        
        // Benchmark the basic convolution
        double basicTime = benchmarkConvolution(
            applyConvolution,
            inputImage,
            outputImageBasic.data(),
            GAUSSIAN_BLUR_3x3,
            filterWidth,
            width,
            height,
            channels
        );
        
        // Benchmark the optimized convolution
        double optimizedTime = benchmarkConvolution(
            applyConvolutionOptimized,
            inputImage,
            outputImageOptimized.data(),
            GAUSSIAN_BLUR_3x3,
            filterWidth,
            width,
            height,
            channels
        );
        
        // Print benchmark results
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Basic convolution:     " << basicTime * 1000 << " ms" << std::endl;
        std::cout << "Optimized convolution: " << optimizedTime * 1000 << " ms" << std::endl;
        std::cout << "Speedup:               " << basicTime / optimizedTime << "x" << std::endl;
        
        // Apply edge detection using the optimized version
        std::cout << "Applying edge detection filter..." << std::endl;
        applyConvolutionOptimized(
            inputImage,
            outputImageEdge.data(),
            EDGE_DETECTION_3x3,
            filterWidth,
            width,
            height,
            channels
        );
        
        // Apply Sobel X filter
        std::cout << "Applying Sobel X filter..." << std::endl;
        applyConvolutionOptimized(
            inputImage,
            outputImageSobelX.data(),
            SOBEL_X_3x3,
            filterWidth,
            width,
            height,
            channels
        );
        
        // Apply Sobel Y filter
        std::cout << "Applying Sobel Y filter..." << std::endl;
        applyConvolutionOptimized(
            inputImage,
            outputImageSobelY.data(),
            SOBEL_Y_3x3,
            filterWidth,
            width,
            height,
            channels
        );
        
        // Save the processed images
        std::cout << "Saving processed images..." << std::endl;
        saveJpg("blurred_basic.jpg", outputImageBasic.data(), width, height, channels);
        saveJpg("blurred_optimized.jpg", outputImageOptimized.data(), width, height, channels);
        saveJpg("edges.jpg", outputImageEdge.data(), width, height, channels);
        saveJpg("sobel_x.jpg", outputImageSobelX.data(), width, height, channels);
        saveJpg("sobel_y.jpg", outputImageSobelY.data(), width, height, channels);
        
        std::cout << "Images saved as:" << std::endl;
        std::cout << "  - original.jpg" << std::endl;
        std::cout << "  - blurred_basic.jpg" << std::endl;
        std::cout << "  - blurred_optimized.jpg" << std::endl;
        std::cout << "  - edges.jpg" << std::endl;
        std::cout << "  - sobel_x.jpg" << std::endl;
        std::cout << "  - sobel_y.jpg" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        
        // Free resources
        if (useTestImage && inputImage) {
            delete[] inputImage;
        } else if (inputImage) {
            stbi_image_free(inputImage);
        }
        
        return 1;
    }
    
    // Free resources
    if (useTestImage && inputImage) {
        delete[] inputImage;
    } else if (inputImage) {
        stbi_image_free(inputImage);
    }
    
    return 0;
}