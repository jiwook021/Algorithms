/**
 * @file test_OrbSlam.cpp
 * @brief Unit tests for OrbSlam module
 */

#include "OrbSlam.hpp"
#include <cassert>
#include <iostream>

namespace {

void TestDetectCornersOnCheckerboard() {
    // Create a simple checkerboard pattern (strong corners expected)
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC1);
    for (int y = 0; y < 100; y += 20)
        for (int x = 0; x < 100; x += 20)
            cv::rectangle(img, cv::Rect(x, y, 10, 10), cv::Scalar(255), -1);

    std::vector<cv::Point> corners;
    OrbSlam::DetectHarrisCorners(img, corners);
    assert(!corners.empty());
    std::cout << "  TestDetectCornersOnCheckerboard PASSED (" << corners.size() << " corners)\n";
}

void TestNoCornersOnBlankImage() {
    cv::Mat blank(50, 50, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Point> corners;
    OrbSlam::DetectHarrisCorners(blank, corners);
    assert(corners.empty());
    std::cout << "  TestNoCornersOnBlankImage PASSED\n";
}

void TestDescriptorDimensions() {
    cv::Mat img(50, 50, CV_8UC1, cv::Scalar(100));
    // Place a corner-like structure
    cv::rectangle(img, cv::Rect(20, 20, 15, 15), cv::Scalar(255), -1);

    std::vector<cv::Point> corners;
    OrbSlam::DetectHarrisCorners(img, corners);

    cv::Mat desc;
    OrbSlam::ComputeSimpleDescriptors(img, corners, desc, 5);

    if (!corners.empty()) {
        assert(desc.rows == static_cast<int>(corners.size()));
        assert(desc.cols == 25);  // 5x5 flattened
    }
    std::cout << "  TestDescriptorDimensions PASSED\n";
}

void TestDescriptorValues() {
    // Uniform image -- all descriptor values should match the pixel value
    cv::Mat img(30, 30, CV_8UC1, cv::Scalar(42));
    std::vector<cv::Point> corners = {{15, 15}};
    cv::Mat desc;
    OrbSlam::ComputeSimpleDescriptors(img, corners, desc, 3);

    for (int c = 0; c < desc.cols; ++c)
        assert(desc.at<float>(0, c) == 42.0f);
    std::cout << "  TestDescriptorValues PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running OrbSlam tests...\n";
    TestDetectCornersOnCheckerboard();
    TestNoCornersOnBlankImage();
    TestDescriptorDimensions();
    TestDescriptorValues();
    std::cout << "All OrbSlam tests PASSED.\n";
    return 0;
}
