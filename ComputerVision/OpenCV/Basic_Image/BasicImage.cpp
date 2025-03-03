/**
 * @file BasicImage.cpp
 * @brief Implementation of basic image operations
 */

#include "BasicImage.hpp"
#include <iostream>

namespace BasicImage {

cv::Mat LoadImage(const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (img.empty())
        std::cerr << "Error: Could not load " << path << std::endl;
    return img;
}

cv::Vec3b GetPixel(const cv::Mat& image, int row, int col) {
    return image.at<cv::Vec3b>(row, col);
}

void SetPixel(cv::Mat& image, int row, int col, const cv::Vec3b& color) {
    image.at<cv::Vec3b>(row, col) = color;
}

cv::Mat ExtractRoi(const cv::Mat& image, const cv::Rect& roi) {
    return image(roi).clone();
}

cv::Mat ConvertToGray(const cv::Mat& bgrImage) {
    cv::Mat gray;
    cv::cvtColor(bgrImage, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

}  // namespace BasicImage
