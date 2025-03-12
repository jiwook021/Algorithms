/**
 * @file ImageSegmentation.cu
 * @brief CUDA implementation of K-means clustering for image segmentation using OpenCV.
 * 
 * This program performs image segmentation using the K-means clustering algorithm
 * accelerated by CUDA on the GPU. It processes an input image and generates a
 * segmented version where pixels are colored according to their cluster assignment.
 * 
 * Time Complexity:
 * - O(n * k * i) where n is the number of pixels, k is the number of clusters, and i is the number of iterations.
 * - Parallelization on the GPU reduces this to approximately O(n/p * k * i) where p is the number of parallel threads.
 * 
 * Space Complexity:
 * - O(n + k) where n is the number of pixels and k is the number of clusters.
 * - GPU memory usage: O(n + k) for storing pixels, assignments, and centroids.
 * 
 * Compilation: nvcc -O2 --expt-relaxed-constexpr main.cu -o main -ccbin g++-10 -I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_features2d
 * Usage: ./main [options]
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include <iostream>
 #include <vector>
 #include <algorithm>
 #include <cmath>
 #include <string.h>
 #include <chrono>
 #include <cassert>
 
 // Include OpenCV libraries for image processing
 #include <opencv2/core.hpp>
 #include <opencv2/imgcodecs.hpp>
 #include <opencv2/highgui.hpp>
 #include <opencv2/imgproc.hpp>
 
 /**
  * @brief Error checking macro for CUDA calls
  * 
  * This macro wraps CUDA API calls and checks for errors.
  * If an error occurs, it prints the error message and exits the program.
  */
 #define CUDA_CHECK(call) \
     do { \
         cudaError_t error = call; \
         if (error != cudaSuccess) { \
             fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(error)); \
             exit(EXIT_FAILURE); \
         } \
     } while(0)
 
 /**
  * @brief Struct to represent a pixel in RGB space
  */
 struct Pixel {
     unsigned char r, g, b;
 };
 
 /**
  * @brief Struct to represent a cluster centroid
  * 
  * Uses floating-point values for more accurate calculations during updates.
  * Includes a count field for the number of pixels assigned to this cluster.
  */
 struct Centroid {
     float r, g, b;
     int count;
 };
 
 /**
  * @brief K-means cluster assignment kernel
  * 
  * This kernel assigns each pixel to the nearest centroid based on Euclidean distance in RGB space.
  * Each thread processes one pixel, making this highly parallelizable.
  * 
  * @param pixels Array of input pixels
  * @param assignments Output array to store cluster assignments for each pixel
  * @param centroids Array of current cluster centroids
  * @param width Image width
  * @param height Image height
  * @param k Number of clusters
  */
 __global__ void assignClusters(const Pixel* pixels, int* assignments, const Centroid* centroids, 
                               int width, int height, int k) {
     // Calculate global thread position
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     
     // Check if this thread is within image bounds
     if (x < width && y < height) {
         int idx = y * width + x;
         const Pixel& pixel = pixels[idx];
         
         int nearestCluster = 0;
         float minDistance = INFINITY;
         
         // Find the nearest centroid to this pixel
         for (int i = 0; i < k; i++) {
             // Calculate Euclidean distance in RGB space
             float dr = pixel.r - centroids[i].r;
             float dg = pixel.g - centroids[i].g;
             float db = pixel.b - centroids[i].b;
             float distance = dr*dr + dg*dg + db*db;
             
             if (distance < minDistance) {
                 minDistance = distance;
                 nearestCluster = i;
             }
         }
         
         // Assign this pixel to the nearest cluster
         assignments[idx] = nearestCluster;
     }
 }
 
 /**
  * @brief K-means centroid update kernel (accumulation step)
  * 
  * This kernel accumulates the RGB values of all pixels assigned to each cluster.
  * Each thread processes one pixel and updates the corresponding cluster's accumulators.
  * Uses atomic operations to safely update from multiple threads.
  * 
  * @param pixels Array of input pixels
  * @param assignments Array of cluster assignments for each pixel
  * @param newCentroids Output array to accumulate values for new centroids
  * @param width Image width
  * @param height Image height
  * @param k Number of clusters
  */
 __global__ void updateCentroidAccumulate(const Pixel* pixels, const int* assignments, 
                                       Centroid* newCentroids, int width, int height, int k) {
     // Calculate global thread position
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     
     // Check if this thread is within image bounds
     if (x < width && y < height) {
         int idx = y * width + x;
         int clusterIdx = assignments[idx];
         
         // Use atomics to safely update cluster accumulators from multiple threads
         // This is necessary because multiple threads may try to update the same centroid simultaneously
         atomicAdd(&newCentroids[clusterIdx].r, pixels[idx].r);
         atomicAdd(&newCentroids[clusterIdx].g, pixels[idx].g);
         atomicAdd(&newCentroids[clusterIdx].b, pixels[idx].b);
         atomicAdd(&newCentroids[clusterIdx].count, 1);
     }
 }
 
 /**
  * @brief K-means centroid normalization kernel
  * 
  * This kernel calculates the average RGB values for each cluster based on accumulated values.
  * Each thread processes one cluster centroid.
  * 
  * @param newCentroids Array of accumulated values for new centroids
  * @param centroids Output array to store normalized centroid values
  * @param k Number of clusters
  */
 __global__ void normalizeCentroids(Centroid* newCentroids, Centroid* centroids, int k) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     
     // Check if this thread is within the range of clusters
     if (i < k) {
         // Calculate average values only if cluster has assigned pixels
         if (newCentroids[i].count > 0) {
             centroids[i].r = newCentroids[i].r / newCentroids[i].count;
             centroids[i].g = newCentroids[i].g / newCentroids[i].count;
             centroids[i].b = newCentroids[i].b / newCentroids[i].count;
         }
         // Reset accumulators for next iteration
         newCentroids[i].r = 0.0f;
         newCentroids[i].g = 0.0f; 
         newCentroids[i].b = 0.0f;
         newCentroids[i].count = 0;
     }
 }
 
 /**
  * @brief Kernel to color each pixel according to its centroid
  * 
  * This kernel assigns the color of each pixel's centroid to the output image.
  * Each thread processes one pixel.
  * 
  * @param output Output pixel array
  * @param input Input pixel array
  * @param assignments Array of cluster assignments for each pixel
  * @param centroids Array of cluster centroids
  * @param width Image width
  * @param height Image height
  */
 __global__ void colorPixels(Pixel* output, const Pixel* input, const int* assignments, 
                             const Centroid* centroids, int width, int height) {
     // Calculate global thread position
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     
     // Check if this thread is within image bounds
     if (x < width && y < height) {
         int idx = y * width + x;
         int clusterIdx = assignments[idx];
         
         // Set output pixel color to its centroid color
         output[idx].r = (unsigned char)centroids[clusterIdx].r;
         output[idx].g = (unsigned char)centroids[clusterIdx].g;
         output[idx].b = (unsigned char)centroids[clusterIdx].b;
     }
 }
 
 /**
  * @brief Initialize cluster centroids using the k-means++ method
  * 
  * K-means++ is an initialization method that chooses centroids that are far apart from each other.
  * This generally leads to better and faster convergence than random initialization.
  * 
  * Time Complexity: O(n * k) where n is the number of pixels and k is the number of clusters.
  * 
  * @param pixels Vector of input pixels
  * @param centroids Output vector to store initialized centroids
  * @param k Number of clusters
  */
 void initializeCentroidsPlusPlus(const std::vector<Pixel>& pixels, std::vector<Centroid>& centroids, int k) {
     int n = pixels.size();
     std::vector<float> distances(n, INFINITY);
     
     // Choose first centroid randomly
     int firstIdx = rand() % n;
     centroids[0].r = pixels[firstIdx].r;
     centroids[0].g = pixels[firstIdx].g;
     centroids[0].b = pixels[firstIdx].b;
     centroids[0].count = 0;
     
     // Choose subsequent centroids
     for (int i = 1; i < k; i++) {
         float sum = 0.0f;
         
         // Update distances to nearest centroid
         for (int j = 0; j < n; j++) {
             float minDist = INFINITY;
             for (int c = 0; c < i; c++) {
                 float dr = pixels[j].r - centroids[c].r;
                 float dg = pixels[j].g - centroids[c].g;
                 float db = pixels[j].b - centroids[c].b;
                 float dist = dr*dr + dg*dg + db*db;
                 minDist = std::min(minDist, dist);
             }
             distances[j] = minDist;
             sum += distances[j];
         }
         
         // Select next centroid with probability proportional to distance squared
         float threshold = sum * static_cast<float>(rand()) / RAND_MAX;
         sum = 0.0f;
         int nextIdx = 0;
         for (int j = 0; j < n; j++) {
             sum += distances[j];
             if (sum >= threshold) {
                 nextIdx = j;
                 break;
             }
         }
         
         centroids[i].r = pixels[nextIdx].r;
         centroids[i].g = pixels[nextIdx].g;
         centroids[i].b = pixels[nextIdx].b;
         centroids[i].count = 0;
     }
 }
 
 /**
  * @brief Main K-means clustering function for image segmentation using CUDA
  * 
  * This function performs the following steps:
  * 1. Load the input image
  * 2. Initialize cluster centroids using k-means++
  * 3. Allocate memory on the GPU
  * 4. Run the K-means algorithm on the GPU
  * 5. Generate the segmented output image
  * 6. Save the result
  * 
  * Time Complexity: O(n * k * i) where n is the number of pixels, k is the number of clusters,
  *                  and i is the number of iterations. With GPU parallelization, the effective
  *                  complexity is reduced significantly.
  * 
  * Space Complexity: O(n + k) on both host and device memory.
  * 
  * @param inputPath Path to the input image file
  * @param outputPath Path to save the output segmented image
  * @param k Number of clusters (segments)
  * @param maxIterations Maximum number of K-means iterations
  * @param convergenceThreshold Threshold for early stopping based on intra-cluster distance change
  */
 void segmentImageWithKMeansCUDA(const char* inputPath, const char* outputPath, int k, int maxIterations, float convergenceThreshold) {
     // Start timing
     auto startTime = std::chrono::high_resolution_clock::now();
     
     // ================== Load Image with OpenCV ==================
     cv::Mat image = cv::imread(inputPath);
     if (image.empty()) {
         std::cerr << "Error: Could not load image " << inputPath << std::endl;
         exit(EXIT_FAILURE);
     }
     
     int width = image.cols;
     int height = image.rows;
     std::cout << "Image loaded: " << width << "x" << height << ", " << image.channels() << " channels" << std::endl;
     
     // Convert image to BGR to ensure we have 3 channels
     if (image.channels() != 3) {
         cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
     }
     
     // Convert OpenCV Mat to pixel array
     std::vector<Pixel> pixels(width * height);
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
             // Note: OpenCV uses BGR order
             pixels[y * width + x].b = pixel[0];
             pixels[y * width + x].g = pixel[1];
             pixels[y * width + x].r = pixel[2];
         }
     }
     
     // ================== Initialize Centroids ==================
     // Seed random number generator for reproducibility
     srand(42);
     
     // Initialize cluster centroids using k-means++
     std::vector<Centroid> centroids(k);
     initializeCentroidsPlusPlus(pixels, centroids, k);
     
     // ================== Allocate GPU Memory ==================
     Pixel* d_pixels;
     Pixel* d_output;
     int* d_assignments;
     Centroid* d_centroids;
     Centroid* d_newCentroids;
     
     size_t pixelsSize = width * height * sizeof(Pixel);
     size_t assignmentsSize = width * height * sizeof(int);
     size_t centroidsSize = k * sizeof(Centroid);
     
     // Allocate device memory
     CUDA_CHECK(cudaMalloc(&d_pixels, pixelsSize));
     CUDA_CHECK(cudaMalloc(&d_output, pixelsSize));
     CUDA_CHECK(cudaMalloc(&d_assignments, assignmentsSize));
     CUDA_CHECK(cudaMalloc(&d_centroids, centroidsSize));
     CUDA_CHECK(cudaMalloc(&d_newCentroids, centroidsSize));
     
     // Copy data to the device
     CUDA_CHECK(cudaMemcpy(d_pixels, pixels.data(), pixelsSize, cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_centroids, centroids.data(), centroidsSize, cudaMemcpyHostToDevice));
     
     // Initialize new centroids to zeros
     Centroid* initialNewCentroids = new Centroid[k];
     for (int i = 0; i < k; i++) {
         initialNewCentroids[i] = {0.0f, 0.0f, 0.0f, 0};
     }
     CUDA_CHECK(cudaMemcpy(d_newCentroids, initialNewCentroids, centroidsSize, cudaMemcpyHostToDevice));
     delete[] initialNewCentroids;
     
     // ================== Setup Execution Configuration ==================
     // Choose block size for optimal GPU utilization
     dim3 blockSize(16, 16);
     dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                   (height + blockSize.y - 1) / blockSize.y);
     
     // ================== Main K-means Loop ==================
     float prevIntraClusterDistance = INFINITY;
     bool converged = false;
     
     std::cout << "Starting K-means clustering with k=" << k << ", max iterations=" << maxIterations << std::endl;
     
     for (int iter = 0; iter < maxIterations && !converged; iter++) {
         // Assign each pixel to nearest centroid
         assignClusters<<<gridSize, blockSize>>>(d_pixels, d_assignments, d_centroids, width, height, k);
         CUDA_CHECK(cudaGetLastError());
         
         // Update centroids based on assignments
         updateCentroidAccumulate<<<gridSize, blockSize>>>(d_pixels, d_assignments, d_newCentroids, width, height, k);
         CUDA_CHECK(cudaGetLastError());
         
         // Normalize the new centroids
         normalizeCentroids<<<(k + 255) / 256, 256>>>(d_newCentroids, d_centroids, k);
         CUDA_CHECK(cudaGetLastError());
         
         // Check for convergence every few iterations to reduce overhead
         if (iter % 5 == 0 || iter == maxIterations - 1) {
             // Copy centroids back to host to check convergence
             CUDA_CHECK(cudaMemcpy(centroids.data(), d_centroids, centroidsSize, cudaMemcpyDeviceToHost));
             
             // Calculate intra-cluster distance (sum of squared distances to centroids)
             float intraClusterDistance = 0.0f;
             std::vector<int> assignments(width * height);
             CUDA_CHECK(cudaMemcpy(assignments.data(), d_assignments, assignmentsSize, cudaMemcpyDeviceToHost));
             
             for (int i = 0; i < width * height; i++) {
                 int clusterId = assignments[i];
                 float dr = pixels[i].r - centroids[clusterId].r;
                 float dg = pixels[i].g - centroids[clusterId].g;
                 float db = pixels[i].b - centroids[clusterId].b;
                 intraClusterDistance += dr*dr + dg*dg + db*db;
             }
             
             // Check if converged (relative change in intra-cluster distance is small)
             float change = std::abs(prevIntraClusterDistance - intraClusterDistance) / prevIntraClusterDistance;
             if (change < convergenceThreshold && iter > 0) {
                 converged = true;
                 std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
             }
             
             prevIntraClusterDistance = intraClusterDistance;
             std::cout << "Iteration " << iter + 1 << ": Intra-cluster distance = " << intraClusterDistance << std::endl;
         }
     }
     
     // ================== Generate Output Image ==================
     // Color each pixel according to its centroid
     colorPixels<<<gridSize, blockSize>>>(d_output, d_pixels, d_assignments, d_centroids, width, height);
     CUDA_CHECK(cudaGetLastError());
     
     // Copy result back to host
     std::vector<Pixel> outputPixels(width * height);
     CUDA_CHECK(cudaMemcpy(outputPixels.data(), d_output, pixelsSize, cudaMemcpyDeviceToHost));
     
     // Convert result to OpenCV Mat
     cv::Mat outputImage(height, width, CV_8UC3);
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             int idx = y * width + x;
             // OpenCV uses BGR order
             outputImage.at<cv::Vec3b>(y, x)[0] = outputPixels[idx].b;
             outputImage.at<cv::Vec3b>(y, x)[1] = outputPixels[idx].g;
             outputImage.at<cv::Vec3b>(y, x)[2] = outputPixels[idx].r;
         }
     }
     
     // Save the result
     bool success = cv::imwrite(outputPath, outputImage);
     if (!success) {
         std::cerr << "Error: Could not save output image " << outputPath << std::endl;
     } else {
         std::cout << "Segmented image saved to " << outputPath << std::endl;
     }
     
     // ================== Free Resources ==================
     CUDA_CHECK(cudaFree(d_pixels));
     CUDA_CHECK(cudaFree(d_output));
     CUDA_CHECK(cudaFree(d_assignments));
     CUDA_CHECK(cudaFree(d_centroids));
     CUDA_CHECK(cudaFree(d_newCentroids));
     
     // Print execution time
     auto endTime = std::chrono::high_resolution_clock::now();
     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
     std::cout << "Total execution time: " << duration << " ms" << std::endl;
 }
 
 /**
  * @brief Main function
  * 
  * Parses command line arguments and runs the image segmentation.
  * 
  * @param argc Number of command line arguments
  * @param argv Array of command line arguments
  * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
  */
 int main(int argc, char** argv) {
     // Default parameters
     const char* inputPath = "input.png";
     const char* outputPath = "segmented.png";
     int k = 5; // Number of clusters
     int maxIterations = 100;
     float convergenceThreshold = 0.01f;
     
     // Parse command line arguments if provided
     for (int i = 1; i < argc; i++) {
         if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
             inputPath = argv[i + 1];
             i++;
         } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
             outputPath = argv[i + 1];
             i++;
         } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
             k = std::max(2, std::min(255, atoi(argv[i + 1]))); // Limit k to reasonable range
             i++;
         } else if (strcmp(argv[i], "-iter") == 0 && i + 1 < argc) {
             maxIterations = std::max(1, atoi(argv[i + 1]));
             i++;
         } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
             // Fix: Convert to float explicitly and use same type in min/max
             float threshold = static_cast<float>(atof(argv[i + 1]));
             convergenceThreshold = std::max(0.0001f, std::min(0.1f, threshold));
             i++;
         } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
             std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
             std::cout << "Options:" << std::endl;
             std::cout << "  -i <path>     Input image path (default: input.png)" << std::endl;
             std::cout << "  -o <path>     Output image path (default: segmented.png)" << std::endl;
             std::cout << "  -k <number>   Number of clusters (default: 5)" << std::endl;
             std::cout << "  -iter <number> Maximum iterations (default: 100)" << std::endl;
             std::cout << "  -t <number>   Convergence threshold (default: 0.01)" << std::endl;
             std::cout << "  -h, --help    Show this help message" << std::endl;
             return EXIT_SUCCESS;
         }
     }
     
     // Print parameters
     std::cout << "Image Segmentation using GPU-based K-means Clustering" << std::endl;
     std::cout << "------------------------------------------------" << std::endl;
     std::cout << "Input:  " << inputPath << std::endl;
     std::cout << "Output: " << outputPath << std::endl;
     std::cout << "K:      " << k << std::endl;
     std::cout << "Max iterations: " << maxIterations << std::endl;
     std::cout << "Convergence threshold: " << convergenceThreshold << std::endl;
     std::cout << "------------------------------------------------" << std::endl;
     
     // Run the segmentation
     segmentImageWithKMeansCUDA(inputPath, outputPath, k, maxIterations, convergenceThreshold);
     
     return EXIT_SUCCESS;
 }