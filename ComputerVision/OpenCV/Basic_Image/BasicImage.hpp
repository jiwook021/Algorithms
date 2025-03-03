/**
 * @file BasicImage.hpp
 * @brief Basic OpenCV image operations: load, pixel access, ROI, color conversion
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace BasicImage {

/// Load an image from file. Returns empty Mat on failure.
cv::Mat LoadImage(const std::string& path);

/// Get pixel value at (row, col) from a BGR image.
cv::Vec3b GetPixel(const cv::Mat& image, int row, int col);

/// Set pixel value at (row, col) in a BGR image.
void SetPixel(cv::Mat& image, int row, int col, const cv::Vec3b& color);

/// Extract a region of interest.
cv::Mat ExtractRoi(const cv::Mat& image, const cv::Rect& roi);

/// Convert BGR to grayscale.
cv::Mat ConvertToGray(const cv::Mat& bgrImage);

}  // namespace BasicImage
