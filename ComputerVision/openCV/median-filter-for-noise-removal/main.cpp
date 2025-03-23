/**
 * @file median_filter.cpp
 * @brief Complete implementation of median filter with Google Test
 * 
 * This file contains both the implementation of a median filter algorithm
 * and Google Test cases to verify its correctness.
 */

 #include <iostream>
 #include <vector>
 #include <algorithm>
 #include <cmath>
 #include <stdexcept>
 #include <string>
 #include <opencv2/opencv.hpp>
 #include <gtest/gtest.h>
 
 using namespace cv;
 using namespace std;
 
 /**
  * @brief Apply a median filter to a grayscale image
  * 
  * This function applies a median filter to remove noise from an image while
  * preserving edges. The filter replaces each pixel with the median value from
  * its neighborhood defined by the kernel size.
  * 
  * Time Complexity: O(rows * cols * kernelSize^2 * log(kernelSize^2))
  * Space Complexity: O(kernelSize^2)
  * 
  * @param inputImage The input grayscale image
  * @param kernelSize The size of the filter kernel (must be odd)
  * @return cv::Mat The filtered image
  * @throws std::invalid_argument If kernel size is not odd or input is invalid
  */
 Mat applyMedianFilter(const Mat& inputImage, int kernelSize) {
     // Validate input parameters
     if (inputImage.empty()) {
         throw std::invalid_argument("Input image is empty");
     }
     
     // Ensure the kernel size is odd
     if (kernelSize % 2 == 0) {
         throw std::invalid_argument("Kernel size must be odd");
     }
     
     if (kernelSize <= 1) {
         throw std::invalid_argument("Kernel size must be greater than 1");
     }
 
     // Get the dimensions of the input image
     int rows = inputImage.rows;
     int cols = inputImage.cols;
 
     // Create an output image with the same size as the input
     Mat outputImage = inputImage.clone();
 
     // Calculate the padding size
     int pad = kernelSize / 2;
 
     // Iterate over each pixel in the image
     for (int i = 0; i < rows; ++i) {
         for (int j = 0; j < cols; ++j) {
             // Extract the neighborhood around the current pixel
             std::vector<uchar> neighborhood;
             neighborhood.reserve(kernelSize * kernelSize);
             
             for (int ki = -pad; ki <= pad; ++ki) {
                 for (int kj = -pad; kj <= pad; ++kj) {
                     // Handle border cases with reflection
                     int ni = std::min(std::max(i + ki, 0), rows - 1);
                     int nj = std::min(std::max(j + kj, 0), cols - 1);
                     
                     neighborhood.push_back(inputImage.at<uchar>(ni, nj));
                 }
             }
 
             // Sort the neighborhood to find the median
             std::sort(neighborhood.begin(), neighborhood.end());
 
             // Replace the current pixel with the median value
             outputImage.at<uchar>(i, j) = neighborhood[neighborhood.size() / 2];
         }
     }
 
     return outputImage;
 }
 
 /**
  * @brief Main application to apply median filter to an image
  * 
  * This function provides a simple command-line interface to apply
  * the median filter to an image file.
  * 
  * @param argc Number of command-line arguments
  * @param argv Command-line arguments (input image, kernel size, output image)
  * @return int Status code (0 for success, non-zero for failure)
  */
 int runMedianFilter(int argc, char** argv) {
     try {
         std::string inputPath = "input_image.png";
         std::string outputPath = "filtered_image.png";
         int kernelSize = 3;
         
         // Parse command-line arguments if provided
         if (argc > 1) inputPath = argv[1];
         if (argc > 2) kernelSize = std::stoi(argv[2]);
         if (argc > 3) outputPath = argv[3];
         
         // Load the grayscale image
         Mat image = imread(inputPath, IMREAD_GRAYSCALE);
 
         if (image.empty()) {
             std::cerr << "Error: Could not open or find the image: " << inputPath << std::endl;
             return 1;
         }
 
         std::cout << "Image loaded successfully. Dimensions: " 
                   << image.cols << "x" << image.rows << std::endl;
         std::cout << "Applying median filter with kernel size: " << kernelSize << "x" << kernelSize << std::endl;
 
         // Apply the median filter
         Mat filteredImage = applyMedianFilter(image, kernelSize);
 
         // Save the filtered image
         imwrite(outputPath, filteredImage);
         std::cout << "Filtered image saved to: " << outputPath << std::endl;
 
         // Display the original and filtered images
         namedWindow("Original Image", WINDOW_AUTOSIZE);
         namedWindow("Filtered Image", WINDOW_AUTOSIZE);
 
         imshow("Original Image", image);
         imshow("Filtered Image", filteredImage);
 
         std::cout << "Press any key to exit..." << std::endl;
         waitKey(0);
 
         return 0;
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
 }
 
 //------------------------------------------------------------------------------
 // Google Test Cases
 //------------------------------------------------------------------------------
 
 /**
  * @brief Test fixture for median filter tests
  */
 class MedianFilterTest : public ::testing::Test {
 protected:
     // Setup function that runs before each test
     void SetUp() override {
         // Create a 5x5 test image with known values
         testImage = Mat(5, 5, CV_8UC1);
         
         // Fill with pattern values
         uchar vals[5][5] = {
             {10, 20, 30, 40, 50},
             {20, 30, 40, 50, 60},
             {30, 40, 50, 60, 70},
             {40, 50, 60, 70, 80},
             {50, 60, 70, 80, 90}
         };
         
         for (int i = 0; i < 5; i++) {
             for (int j = 0; j < 5; j++) {
                 testImage.at<uchar>(i, j) = vals[i][j];
             }
         }
         
         // Create a noisy image for testing noise removal
         noisyImage = testImage.clone();
         
         // Add salt and pepper noise
         noisyImage.at<uchar>(1, 1) = 255; // Salt noise
         noisyImage.at<uchar>(2, 2) = 0;   // Pepper noise
         noisyImage.at<uchar>(3, 3) = 255; // Salt noise
     }
 
     Mat testImage;     // Regular test image
     Mat noisyImage;    // Test image with artificial noise
 };
 
 /**
  * @brief Test invalid kernel sizes
  */
 TEST_F(MedianFilterTest, InvalidKernelSize) {
     // Test even kernel size
     EXPECT_THROW(applyMedianFilter(testImage, 2), std::invalid_argument);
     
     // Test zero kernel size
     EXPECT_THROW(applyMedianFilter(testImage, 0), std::invalid_argument);
     
     // Test negative kernel size
     EXPECT_THROW(applyMedianFilter(testImage, -3), std::invalid_argument);
 }
 
 /**
  * @brief Test with empty image
  */
 TEST_F(MedianFilterTest, EmptyImage) {
     Mat emptyImage;
     EXPECT_THROW(applyMedianFilter(emptyImage, 3), std::invalid_argument);
 }
 
 /**
  * @brief Test that kernel size 3 correctly finds the median
  */
 TEST_F(MedianFilterTest, KernelSize3) {
     // Apply median filter with kernel size 3
     Mat result = applyMedianFilter(testImage, 3);
     
     // Check dimensions are preserved
     EXPECT_EQ(testImage.rows, result.rows);
     EXPECT_EQ(testImage.cols, result.cols);
     
     // For a 3x3 kernel at position (2,2), the median should be 50
     // The 3x3 neighborhood around (2,2) is:
     // 30, 40, 50
     // 40, 50, 60
     // 50, 60, 70
     // Sorted: 30, 40, 40, 50, 50, 60, 60, 70, 70 -> median is 50
     EXPECT_EQ(50, result.at<uchar>(2, 2));
 }
 
 /**
  * @brief Test noise removal capability
  */
 TEST_F(MedianFilterTest, NoiseRemoval) {
     // Apply median filter to noisy image
     Mat result = applyMedianFilter(noisyImage, 3);
     
     // Check that salt and pepper noise is removed
     // At position (1,1) we added salt noise (255), should be closer to original
     EXPECT_NE(255, result.at<uchar>(1, 1));
     EXPECT_NEAR(testImage.at<uchar>(1, 1), result.at<uchar>(1, 1), 20);
     
     // At position (2,2) we added pepper noise (0), should be closer to original
     EXPECT_NE(0, result.at<uchar>(2, 2));
     EXPECT_NEAR(testImage.at<uchar>(2, 2), result.at<uchar>(2, 2), 20);
 }
 
 /**
  * @brief Test that kernel size 5 correctly finds the median
  */
 TEST_F(MedianFilterTest, KernelSize5) {
     // Apply median filter with kernel size 5
     Mat result = applyMedianFilter(testImage, 5);
     
     // Check dimensions are preserved
     EXPECT_EQ(testImage.rows, result.rows);
     EXPECT_EQ(testImage.cols, result.cols);
     
     // For a 5x5 kernel centered at (2,2), the neighborhood includes the entire test image
     // The median value in our 5x5 test image should be 50
     EXPECT_EQ(50, result.at<uchar>(2, 2));
 }
 
 /**
  * @brief Test that border handling works correctly
  */
 TEST_F(MedianFilterTest, BorderHandling) {
     // Apply median filter
     Mat result = applyMedianFilter(testImage, 3);
     
     // Check that border pixels are processed (not left as is)
     EXPECT_NE(testImage.at<uchar>(0, 0), result.at<uchar>(0, 0));
     EXPECT_NE(testImage.at<uchar>(0, 4), result.at<uchar>(0, 4));
     EXPECT_NE(testImage.at<uchar>(4, 0), result.at<uchar>(4, 0));
     EXPECT_NE(testImage.at<uchar>(4, 4), result.at<uchar>(4, 4));
 }
 
 /**
  * @brief Main function
  * 
  * If TESTING is defined, the test cases are run by the Google Test main function.
  * Otherwise, runs the application.
  */
 #ifndef TESTING
 int main(int argc, char** argv) {
     return runMedianFilter(argc, argv);
 }
 #endif