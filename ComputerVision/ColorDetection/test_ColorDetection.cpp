/**
 * @file test_ColorDetection.cpp
 * @brief Unit tests for ColorDetection module
 */

#include "ColorDetection.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

namespace {

/// Create a small synthetic BGR image with a red square in the center.
cv::Mat MakeRedSquareImage(int width, int height, int squareSize) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    int x0 = (width  - squareSize) / 2;
    int y0 = (height - squareSize) / 2;
    // Red in BGR = (0, 0, 255)
    cv::rectangle(img, cv::Rect(x0, y0, squareSize, squareSize),
                  cv::Scalar(0, 0, 255), cv::FILLED);
    return img;
}

void TestConvertToHsv() {
    cv::Mat bgr(4, 4, CV_8UC3, cv::Scalar(255, 0, 0));  // pure blue
    cv::Mat hsv = ColorDetection::ConvertToHsv(bgr);
    assert(hsv.type() == CV_8UC3);
    assert(hsv.rows == 4 && hsv.cols == 4);
    // Blue in HSV: H~120, S=255, V=255
    cv::Vec3b px = hsv.at<cv::Vec3b>(0, 0);
    assert(px[0] == 120);
    std::cout << "  TestConvertToHsv PASSED\n";
}

void TestCreateColorMask() {
    // 4x4 image, top half red, bottom half green
    cv::Mat bgr(4, 4, CV_8UC3, cv::Scalar(0, 255, 0));  // all green
    bgr.rowRange(0, 2).setTo(cv::Scalar(0, 0, 255));     // top = red

    cv::Mat hsv = ColorDetection::ConvertToHsv(bgr);
    ColorDetection::HsvRange redRange{cv::Scalar(0, 100, 100),
                                       cv::Scalar(10, 255, 255)};
    cv::Mat mask = ColorDetection::CreateColorMask(hsv, redRange);

    // Top two rows should be 255, bottom two should be 0
    assert(mask.at<uchar>(0, 0) == 255);
    assert(mask.at<uchar>(1, 0) == 255);
    assert(mask.at<uchar>(2, 0) == 0);
    assert(mask.at<uchar>(3, 0) == 0);
    std::cout << "  TestCreateColorMask PASSED\n";
}

void TestDetectFindsRedSquare() {
    cv::Mat img = MakeRedSquareImage(100, 100, 40);
    ColorDetection::HsvRange redRange{cv::Scalar(0, 100, 100),
                                       cv::Scalar(10, 255, 255)};
    auto results = ColorDetection::Detect(img, redRange, 10, 3);

    assert(!results.empty());
    // Bounding box should roughly cover the 40x40 square in the center
    auto& box = results[0].BoundingBox;
    assert(box.width  >= 35 && box.width  <= 45);
    assert(box.height >= 35 && box.height <= 45);
    std::cout << "  TestDetectFindsRedSquare PASSED\n";
}

void TestDetectIgnoresSmallObjects() {
    cv::Mat img = MakeRedSquareImage(100, 100, 5);  // tiny square
    ColorDetection::HsvRange redRange{cv::Scalar(0, 100, 100),
                                       cv::Scalar(10, 255, 255)};
    auto results = ColorDetection::Detect(img, redRange, 20, 3);
    assert(results.empty());
    std::cout << "  TestDetectIgnoresSmallObjects PASSED\n";
}

void TestCustomInRange() {
    cv::Mat hsv(2, 2, CV_8UC3);
    hsv.at<cv::Vec3b>(0, 0) = {5, 200, 200};   // inside
    hsv.at<cv::Vec3b>(0, 1) = {15, 200, 200};  // outside (H too high)
    hsv.at<cv::Vec3b>(1, 0) = {5, 50, 200};    // outside (S too low)
    hsv.at<cv::Vec3b>(1, 1) = {5, 200, 200};   // inside

    cv::Mat mask = ColorDetection::CustomInRange(
        hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255));

    assert(mask.at<uchar>(0, 0) == 255);
    assert(mask.at<uchar>(0, 1) == 0);
    assert(mask.at<uchar>(1, 0) == 0);
    assert(mask.at<uchar>(1, 1) == 255);
    std::cout << "  TestCustomInRange PASSED\n";
}

void TestCustomBoundingRect() {
    std::vector<cv::Point> contour = {{10, 20}, {30, 20}, {30, 50}, {10, 50}};
    cv::Rect r = ColorDetection::CustomBoundingRect(contour);
    assert(r.x == 10 && r.y == 20);
    assert(r.width == 21 && r.height == 31);
    std::cout << "  TestCustomBoundingRect PASSED\n";
}

void TestCustomFindContours() {
    cv::Mat binary = cv::Mat::zeros(20, 20, CV_8UC1);
    // Draw a 6x6 white block
    binary(cv::Rect(5, 5, 6, 6)).setTo(255);
    auto contours = ColorDetection::CustomFindContours(binary, 5);
    assert(contours.size() == 1);
    assert(static_cast<int>(contours[0].size()) == 36);
    std::cout << "  TestCustomFindContours PASSED\n";
}

void TestCustomMorphOpen() {
    cv::Mat src = cv::Mat::zeros(20, 20, CV_8UC1);
    // Large block
    src(cv::Rect(5, 5, 10, 10)).setTo(255);
    // Tiny noise pixel
    src.at<uchar>(1, 1) = 255;

    cv::Mat kernel = ColorDetection::CustomStructuringElement(cv::Size(3, 3));
    cv::Mat dst;
    ColorDetection::CustomMorphOpen(src, dst, kernel);

    // Noise pixel should be removed by opening
    assert(dst.at<uchar>(1, 1) == 0);
    // Center of the block should remain
    assert(dst.at<uchar>(10, 10) == 255);
    std::cout << "  TestCustomMorphOpen PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running ColorDetection tests...\n";
    TestConvertToHsv();
    TestCreateColorMask();
    TestDetectFindsRedSquare();
    TestDetectIgnoresSmallObjects();
    TestCustomInRange();
    TestCustomBoundingRect();
    TestCustomFindContours();
    TestCustomMorphOpen();
    std::cout << "All ColorDetection tests PASSED.\n";
    return 0;
}
