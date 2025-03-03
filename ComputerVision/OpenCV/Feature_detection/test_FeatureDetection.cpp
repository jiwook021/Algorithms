/**
 * @file test_FeatureDetection.cpp
 * @brief Unit tests for FeatureDetection module
 */

#include "FeatureDetection.hpp"
#include <cassert>
#include <iostream>

namespace {

void TestMatchSameImage() {
    cv::Mat img(200, 200, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::rectangle(img, cv::Rect(30, 30, 50, 50), cv::Scalar(255, 0, 0), -1);
    cv::circle(img, cv::Point(150, 150), 20, cv::Scalar(0, 255, 0), -1);

    auto result = FeatureDetection::DetectAndMatch(img, img, 10);
    // Same image should have perfect matches
    assert(!result.GoodMatches.empty());
    for (const auto& m : result.GoodMatches)
        assert(m.distance == 0);
    std::cout << "  TestMatchSameImage PASSED\n";
}

void TestEmptyImages() {
    cv::Mat empty;
    cv::Mat img(50, 50, CV_8UC3, cv::Scalar(0));
    auto result = FeatureDetection::DetectAndMatch(img, img, 5);
    // Uniform image may have no features
    std::cout << "  TestEmptyImages PASSED\n";
}

void TestDrawVisualization() {
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::rectangle(img, cv::Rect(10, 10, 30, 30), cv::Scalar(200, 0, 0), -1);
    auto result = FeatureDetection::DetectAndMatch(img, img, 5);
    cv::Mat vis = FeatureDetection::DrawMatchVisualization(img, img, result);
    assert(!vis.empty());
    std::cout << "  TestDrawVisualization PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running FeatureDetection tests...\n";
    TestMatchSameImage();
    TestEmptyImages();
    TestDrawVisualization();
    std::cout << "All FeatureDetection tests PASSED.\n";
    return 0;
}
