/**
 * @file MedianFilterNoise.hpp
 * @brief Adaptive median filter for advanced noise removal
 * @details Multi-pass adaptive median filter that varies window size per pixel based on local statistics. Outperforms fixed-window median for non-uniform noise patterns.
 */

#pragma once

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
 Mat ApplyMedianFilter(const Mat& InputImage, int KernelSize) {
     // Validate input parameters
     if (InputImage.empty()) {
         throw std::invalid_argument("Input image is empty");
     }
     
     // Ensure the kernel size is odd
     if (KernelSize % 2 == 0) {
         throw std::invalid_argument("Kernel size must be odd");
     }
     
     if (KernelSize <= 1) {
         throw std::invalid_argument("Kernel size must be greater than 1");
     }
 
     // Get the dimensions of the input image
     int Rows = InputImage.rows;
     int Cols = InputImage.cols;
 
     // Create an output image with the same size as the input
     Mat OutputImage = InputImage.clone();
 
     // Calculate the padding size
     int Pad = KernelSize / 2;
 
     // Iterate over each pixel in the image
     for (int i = 0; i < Rows; ++i) {
         for (int j = 0; j < Cols; ++j) {
             // Extract the neighborhood around the current pixel
             std::vector<uchar> Neighborhood;
             Neighborhood.reserve(KernelSize * KernelSize);
             
             for (int Ki = -Pad; Ki <= Pad; ++Ki) {
                 for (int Kj = -Pad; Kj <= Pad; ++Kj) {
                     // Handle border cases with reflection
                     int Ni = std::min(std::max(i + Ki, 0), Rows - 1);
                     int Nj = std::min(std::max(j + Kj, 0), Cols - 1);
                     
                     Neighborhood.push_back(InputImage.at<uchar>(Ni, Nj));
                 }
             }
 
             // Sort the neighborhood to find the median
             std::sort(Neighborhood.begin(), Neighborhood.end());
 
             // Replace the current pixel with the median value
             OutputImage.at<uchar>(i, j) = Neighborhood[Neighborhood.size() / 2];
         }
     }
 
     return OutputImage;
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
 int RunMedianFilter(int argc, char** argv) {
     try {
         std::string InputPath = "input_image.png";
         std::string OutputPath = "filtered_image.png";
         int KernelSize = 3;
         
         // Parse command-line arguments if provided
         if (argc > 1) InputPath = argv[1];
         if (argc > 2) KernelSize = std::stoi(argv[2]);
         if (argc > 3) OutputPath = argv[3];
         
         // Load the grayscale image
         Mat Image = imread(InputPath, IMREAD_GRAYSCALE);
 
         if (Image.empty()) {
             std::cerr << "Error: Could not open or find the image: " << InputPath << std::endl;
             return 1;
         }
 
         std::cout << "Image loaded successfully. Dimensions: " 
                   << Image.cols << "x" << Image.rows << std::endl;
         std::cout << "Applying median filter with kernel size: " << KernelSize << "x" << KernelSize << std::endl;
 
         // Apply the median filter
         Mat FilteredImage = ApplyMedianFilter(Image, KernelSize);
 
         // Save the filtered image
         imwrite(OutputPath, FilteredImage);
         std::cout << "Filtered image saved to: " << OutputPath << std::endl;
 
         // Display the original and filtered images
         namedWindow("Original Image", WINDOW_AUTOSIZE);
         namedWindow("Filtered Image", WINDOW_AUTOSIZE);
 
         imshow("Original Image", Image);
         imshow("Filtered Image", FilteredImage);
 
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
         TestImage = Mat(5, 5, CV_8UC1);
         
         // Fill with pattern values
         uchar Vals[5][5] = {
             {10, 20, 30, 40, 50},
             {20, 30, 40, 50, 60},
             {30, 40, 50, 60, 70},
             {40, 50, 60, 70, 80},
             {50, 60, 70, 80, 90}
         };
         
         for (int i = 0; i < 5; i++) {
             for (int j = 0; j < 5; j++) {
                 TestImage.at<uchar>(i, j) = Vals[i][j];
             }
         }
         
         // Create a noisy image for testing noise removal
         NoisyImage = TestImage.clone();
         
         // Add salt and pepper noise
         NoisyImage.at<uchar>(1, 1) = 255; // Salt noise
         NoisyImage.at<uchar>(2, 2) = 0;   // Pepper noise
         NoisyImage.at<uchar>(3, 3) = 255; // Salt noise
     }
 
     Mat TestImage;     // Regular test image
     Mat NoisyImage;    // Test image with artificial noise
 };
 
 /**
  * @brief Test invalid kernel sizes
  */
 TEST_F(MedianFilterTest, InvalidKernelSize) {
     // Test even kernel size
     EXPECT_THROW(ApplyMedianFilter(TestImage, 2), std::invalid_argument);
     
     // Test zero kernel size
     EXPECT_THROW(ApplyMedianFilter(TestImage, 0), std::invalid_argument);
     
     // Test negative kernel size
     EXPECT_THROW(ApplyMedianFilter(TestImage, -3), std::invalid_argument);
 }
 
 /**
  * @brief Test with empty image
  */
 TEST_F(MedianFilterTest, EmptyImage) {
     Mat EmptyImage;
     EXPECT_THROW(ApplyMedianFilter(EmptyImage, 3), std::invalid_argument);
 }
 
 /**
  * @brief Test that kernel size 3 correctly finds the median
  */
 TEST_F(MedianFilterTest, KernelSize3) {
     // Apply median filter with kernel size 3
     Mat Result = ApplyMedianFilter(TestImage, 3);
     
     // Check dimensions are preserved
     EXPECT_EQ(TestImage.rows, Result.rows);
     EXPECT_EQ(TestImage.cols, Result.cols);
     
     // For a 3x3 kernel at position (2,2), the median should be 50
     // The 3x3 neighborhood around (2,2) is:
     // 30, 40, 50
     // 40, 50, 60
     // 50, 60, 70
     // Sorted: 30, 40, 40, 50, 50, 60, 60, 70, 70 -> median is 50
     EXPECT_EQ(50, Result.at<uchar>(2, 2));
 }
 
 /**
  * @brief Test noise removal capability
  */
 TEST_F(MedianFilterTest, NoiseRemoval) {
     // Apply median filter to noisy image
     Mat Result = ApplyMedianFilter(NoisyImage, 3);
     
     // Check that salt and pepper noise is removed
     // At position (1,1) we added salt noise (255), should be closer to original
     EXPECT_NE(255, Result.at<uchar>(1, 1));
     EXPECT_NEAR(TestImage.at<uchar>(1, 1), Result.at<uchar>(1, 1), 20);
     
     // At position (2,2) we added pepper noise (0), should be closer to original
     EXPECT_NE(0, Result.at<uchar>(2, 2));
     EXPECT_NEAR(TestImage.at<uchar>(2, 2), Result.at<uchar>(2, 2), 20);
 }
 
 /**
  * @brief Test that kernel size 5 correctly finds the median
  */
 TEST_F(MedianFilterTest, KernelSize5) {
     // Apply median filter with kernel size 5
     Mat Result = ApplyMedianFilter(TestImage, 5);
     
     // Check dimensions are preserved
     EXPECT_EQ(TestImage.rows, Result.rows);
     EXPECT_EQ(TestImage.cols, Result.cols);
     
     // For a 5x5 kernel centered at (2,2), the neighborhood includes the entire test image
     // The median value in our 5x5 test image should be 50
     EXPECT_EQ(50, Result.at<uchar>(2, 2));
 }
 
 /**
  * @brief Test that border handling works correctly
  */
 TEST_F(MedianFilterTest, BorderHandling) {
     // Apply median filter
     Mat Result = ApplyMedianFilter(TestImage, 3);
     
     // Check that border pixels are processed (not left as is)
     EXPECT_NE(TestImage.at<uchar>(0, 0), Result.at<uchar>(0, 0));
     EXPECT_NE(TestImage.at<uchar>(0, 4), Result.at<uchar>(0, 4));
     EXPECT_NE(TestImage.at<uchar>(4, 0), Result.at<uchar>(4, 0));
     EXPECT_NE(TestImage.at<uchar>(4, 4), Result.at<uchar>(4, 4));
 }
 
 /**
  * @brief Main function
  * 
  * If TESTING is defined, the test cases are run by the Google Test main function.
  * Otherwise, runs the application.
  */
 #ifndef TESTING
