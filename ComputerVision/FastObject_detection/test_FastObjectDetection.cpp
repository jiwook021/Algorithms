/**
 * @file test_FastObjectDetection.cpp
 * @brief Unit tests for FastObjectDetection module
 */

#include "FastObjectDetection.hpp"
#include <cassert>
#include <iostream>

namespace {

void TestDetectOnEmptyFrame() {
    FastObjectDetection::FmoDetector detector;
    cv::Mat empty;
    auto results = detector.Detect(empty);
    assert(results.empty());
    std::cout << "  TestDetectOnEmptyFrame PASSED\n";
}

void TestDetectOnBlankFrame() {
    FastObjectDetection::FmoDetector detector;
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto results = detector.Detect(blank);
    // No motion on a single uniform frame
    assert(results.empty());
    std::cout << "  TestDetectOnBlankFrame PASSED\n";
}

void TestDetectOnGrayscaleFrame() {
    FastObjectDetection::FmoDetector detector;
    cv::Mat gray(240, 320, CV_8UC1, cv::Scalar(100));
    auto results = detector.Detect(gray);
    assert(results.empty());
    std::cout << "  TestDetectOnGrayscaleFrame PASSED\n";
}

void TestBackgroundModelBuildsWithoutCrash() {
    FastObjectDetection::FmoDetector detector(30, 50, 5.0f, 5);
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(50, 50, 50));

    // Feed several identical frames -- should build background, no detections
    for (int i = 0; i < 10; ++i) {
        auto results = detector.Detect(frame);
        assert(results.empty());
    }
    std::cout << "  TestBackgroundModelBuildsWithoutCrash PASSED\n";
}

void TestFmoObjectFields() {
    FastObjectDetection::FmoObject fmo;
    fmo.BoundingBox = cv::Rect(10, 20, 30, 40);
    fmo.Direction   = cv::Point2f(1.0f, 0.0f);
    fmo.Speed       = 15.5f;
    fmo.Appearance  = cv::Mat(40, 30, CV_8UC3);

    assert(fmo.BoundingBox.x == 10);
    assert(fmo.Speed > 15.0f);
    assert(fmo.Appearance.rows == 40);
    std::cout << "  TestFmoObjectFields PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running FastObjectDetection tests...\n";
    TestDetectOnEmptyFrame();
    TestDetectOnBlankFrame();
    TestDetectOnGrayscaleFrame();
    TestBackgroundModelBuildsWithoutCrash();
    TestFmoObjectFields();
    std::cout << "All FastObjectDetection tests PASSED.\n";
    return 0;
}
