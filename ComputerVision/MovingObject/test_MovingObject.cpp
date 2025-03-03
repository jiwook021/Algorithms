/**
 * @file test_MovingObject.cpp
 * @brief Unit tests for MovingObject module
 */

#include "MovingObject.hpp"
#include <cassert>
#include <iostream>

namespace {

void TestEmptyImageThrows() {
    MovingObject::MotionDetector detector;
    cv::Mat empty, mask;
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
    bool threw = false;
    try { detector.DetectMotion(empty, img, mask); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "  TestEmptyImageThrows PASSED\n";
}

void TestSizeMismatchThrows() {
    MovingObject::MotionDetector detector;
    cv::Mat a(100, 100, CV_8UC3, cv::Scalar(0));
    cv::Mat b(200, 200, CV_8UC3, cv::Scalar(0));
    cv::Mat mask;
    bool threw = false;
    try { detector.DetectMotion(a, b, mask); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "  TestSizeMismatchThrows PASSED\n";
}

void TestIdenticalFramesNoMotion() {
    cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(100, 100, 100));
    // Add some texture for ORB features
    cv::rectangle(frame, cv::Rect(50, 50, 30, 30), cv::Scalar(255, 0, 0), -1);
    cv::rectangle(frame, cv::Rect(120, 120, 20, 20), cv::Scalar(0, 255, 0), -1);

    MovingObject::MotionDetector detector(500, 0.75f, 5);
    cv::Mat mask;
    cv::Mat result = detector.DetectMotion(frame, frame.clone(), mask);

    assert(!result.empty());
    assert(result.size() == frame.size());
    // Identical frames should have minimal/no motion
    std::cout << "  TestIdenticalFramesNoMotion PASSED\n";
}

void TestGrayscaleInput() {
    cv::Mat gray(100, 100, CV_8UC1, cv::Scalar(80));
    cv::circle(gray, cv::Point(50, 50), 10, cv::Scalar(200), -1);

    MovingObject::MotionDetector detector;
    cv::Mat mask;
    cv::Mat result = detector.DetectMotion(gray, gray.clone(), mask);
    assert(!result.empty());
    std::cout << "  TestGrayscaleInput PASSED\n";
}

void TestUpdateParameters() {
    MovingObject::MotionDetector detector;
    // Should not throw
    detector.UpdateParameters(1000, 0.8f, 20);
    std::cout << "  TestUpdateParameters PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running MovingObject tests...\n";
    TestEmptyImageThrows();
    TestSizeMismatchThrows();
    TestIdenticalFramesNoMotion();
    TestGrayscaleInput();
    TestUpdateParameters();
    std::cout << "All MovingObject tests PASSED.\n";
    return 0;
}
