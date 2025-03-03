/**
 * @file main.cpp
 * @brief Driver for BasicImage module
 */

#include "BasicImage.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "input.jpg";

    cv::Mat image = BasicImage::LoadImage(path);
    if (image.empty()) return -1;

    std::cout << "Dimensions: " << image.cols << "x" << image.rows << "\n"
              << "Channels: " << image.channels() << "\n";

    auto px = BasicImage::GetPixel(image, 100, 100);
    std::cout << "Pixel(100,100): B=" << (int)px[0]
              << " G=" << (int)px[1] << " R=" << (int)px[2] << "\n";

    BasicImage::SetPixel(image, 100, 100, cv::Vec3b(0, 0, 255));

    cv::Mat gray = BasicImage::ConvertToGray(image);
    cv::Mat roi  = BasicImage::ExtractRoi(image, cv::Rect(100, 100, 200, 200));

    cv::imwrite("output_gray.jpg", gray);
    cv::imwrite("output_roi.jpg", roi);

    cv::imshow("Original", image);
    cv::imshow("Gray", gray);
    cv::waitKey(0);
    return 0;
}
