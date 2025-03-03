/**
 * @file test_Scanner.cpp
 * @brief Unit tests for Scanner module
 */

#include "Scanner.hpp"
#include <cassert>
#include <iostream>

namespace {

void TestOrderPoints() {
    std::vector<cv::Point2f> pts = {{100, 0}, {0, 0}, {100, 100}, {0, 100}};
    auto ordered = OrderPoints(pts);
    assert(ordered[0] == cv::Point2f(0, 0));       // top-left
    assert(ordered[2] == cv::Point2f(100, 100));   // bottom-right
    std::cout << "  TestOrderPoints PASSED\n";
}

void TestFourPointTransform() {
    cv::Mat img(200, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::rectangle(img, cv::Rect(50, 50, 100, 100), cv::Scalar(255, 255, 255), -1);

    std::vector<cv::Point2f> pts = {{50, 50}, {150, 50}, {150, 150}, {50, 150}};
    cv::Mat warped = FourPointTransform(img, pts);
    assert(!warped.empty());
    assert(warped.cols > 0 && warped.rows > 0);
    std::cout << "  TestFourPointTransform PASSED\n";
}

void TestEnhanceDocument() {
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::putText(img, "Hi", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 0, 0), 2);
    cv::Mat enhanced = EnhanceDocument(img);
    assert(enhanced.channels() == 1);
    assert(enhanced.size() == img.size());
    std::cout << "  TestEnhanceDocument PASSED\n";
}

void TestRemoveShadows() {
    cv::Mat img(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat result = RemoveShadows(img);
    assert(result.channels() == 3);
    assert(result.size() == img.size());
    std::cout << "  TestRemoveShadows PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running Scanner tests...\n";
    TestOrderPoints();
    TestFourPointTransform();
    TestEnhanceDocument();
    TestRemoveShadows();
    std::cout << "All Scanner tests PASSED.\n";
    return 0;
}
