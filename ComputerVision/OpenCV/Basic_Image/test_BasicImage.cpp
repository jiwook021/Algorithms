/**
 * @file test_BasicImage.cpp
 * @brief Unit tests for BasicImage module
 */

#include "BasicImage.hpp"
#include <cassert>
#include <iostream>

namespace {

void TestGetSetPixel() {
    cv::Mat img(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    BasicImage::SetPixel(img, 5, 5, cv::Vec3b(100, 150, 200));
    auto px = BasicImage::GetPixel(img, 5, 5);
    assert(px[0] == 100 && px[1] == 150 && px[2] == 200);
    std::cout << "  TestGetSetPixel PASSED\n";
}

void TestExtractRoi() {
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(42, 42, 42));
    cv::Mat roi = BasicImage::ExtractRoi(img, cv::Rect(10, 10, 20, 20));
    assert(roi.rows == 20 && roi.cols == 20);
    assert(roi.at<cv::Vec3b>(0, 0) == cv::Vec3b(42, 42, 42));
    std::cout << "  TestExtractRoi PASSED\n";
}

void TestConvertToGray() {
    cv::Mat bgr(5, 5, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat gray = BasicImage::ConvertToGray(bgr);
    assert(gray.channels() == 1);
    assert(gray.rows == 5 && gray.cols == 5);
    assert(gray.at<uchar>(0, 0) == 128);
    std::cout << "  TestConvertToGray PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running BasicImage tests...\n";
    TestGetSetPixel();
    TestExtractRoi();
    TestConvertToGray();
    std::cout << "All BasicImage tests PASSED.\n";
    return 0;
}
