/**
 * @file ImageProcessing.hpp
 * @brief Common OpenCV image processing operations
 */
#pragma once
#include <opencv2/opencv.hpp>
namespace ImageProcessing {
cv::Mat ConvertToGray(const cv::Mat& bgr);
cv::Mat Resize(const cv::Mat& src, int width, int height);
cv::Mat GaussianBlur(const cv::Mat& src, int kernelSize = 5);
cv::Mat Threshold(const cv::Mat& gray, double thresh = 127, double maxVal = 255);
cv::Mat CannyEdge(const cv::Mat& gray, double low = 100, double high = 200);
cv::Mat Rotate(const cv::Mat& src, double angle, double scale = 1.0);
}
