/**
 * @file test_MovingObjectVideo.cpp
 * @brief Unit tests for MovingObjectVideo module
 */

#include "MovingObjectVideo.hpp"
#include <cassert>
#include <iostream>

namespace {

void TestEmptyImageThrows() {
    MovingObjectVideo::MotionDetector detector;
    cv::Mat empty, mask;
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(0));
    bool threw = false;
    try { detector.DetectMotion(empty, img, mask); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "  TestEmptyImageThrows PASSED\n";
}

void TestIdenticalFrames() {
    cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::rectangle(frame, cv::Rect(50, 50, 30, 30), cv::Scalar(255, 0, 0), -1);

    MovingObjectVideo::MotionDetector detector;
    cv::Mat mask;
    cv::Mat result = detector.DetectMotion(frame, frame.clone(), mask);
    assert(!result.empty());
    assert(result.size() == frame.size());
    std::cout << "  TestIdenticalFrames PASSED\n";
}

void TestNegativeTimeThrows() {
    MovingObjectVideo::MotionDetector detector;
    cv::Mat img(50, 50, CV_8UC3, cv::Scalar(0));
    cv::Mat mask;
    bool threw = false;
    try { detector.DetectMotion(img, img.clone(), mask, -1.0f); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "  TestNegativeTimeThrows PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running MovingObjectVideo tests...\n";
    TestEmptyImageThrows();
    TestIdenticalFrames();
    TestNegativeTimeThrows();
    std::cout << "All MovingObjectVideo tests PASSED.\n";
    return 0;
}
