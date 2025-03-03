/**
 * @file SobelEdgeDetection.hpp
 * @brief Sobel edge detection with OpenCV and custom implementations
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace SobelEdgeDetection {

/// OpenCV-based Sobel edge detection pipeline
struct SobelResult {
    cv::Mat GradientX;
    cv::Mat GradientY;
    cv::Mat Combined;
    cv::Mat Thresholded;
};

SobelResult DetectEdges(const cv::Mat& bgrImage, double threshold = 100);

// --- Custom implementations (educational) ---

/// Manual BGR to grayscale using luminance formula
cv::Mat CustomCvtColor(const cv::Mat& bgr);

/// Manual 3x3 Sobel convolution
void CustomSobel(const cv::Mat& gray, cv::Mat& gradX, cv::Mat& gradY);

/// Manual absolute value conversion
cv::Mat CustomConvertScaleAbs(const cv::Mat& src);

/// Manual weighted addition
cv::Mat CustomAddWeighted(const cv::Mat& src1, double alpha,
                          const cv::Mat& src2, double beta, double gamma);

/// Manual binary threshold
cv::Mat CustomThreshold(const cv::Mat& src, double thresh, double maxVal);

}  // namespace SobelEdgeDetection
