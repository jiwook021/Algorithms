/**
 * @file test_OrbSlam.cpp
 * @brief Unit tests for OrbSlam feature matching pipeline.
 *
 * Tests core algorithms with synthetic data (no image files needed).
 */

#include "OrbSlam.hpp"
#include <iostream>
#include <cassert>

static int testsPassed = 0;
static int testsFailed = 0;

void TestHarrisCornersOnCheckerboard() {
    // Create a simple 8x8 checkerboard pattern
    cv::Mat checkerboard(64, 64, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < 64; y++) {
        for (int x = 0; x < 64; x++) {
            if ((x / 8 + y / 8) % 2 == 0) {
                checkerboard.at<uchar>(y, x) = 255;
            }
        }
    }

    std::vector<cv::Point> corners;
    DetectHarrisCorners(checkerboard, corners);

    // Should detect some corners at checkerboard intersections
    if (corners.size() > 0) {
        testsPassed++;
        std::cout << "[PASSED] TestHarrisCornersOnCheckerboard: " << corners.size() << " corners" << std::endl;
    } else {
        testsFailed++;
        std::cerr << "[FAIL] TestHarrisCornersOnCheckerboard: no corners detected" << std::endl;
    }
}

void TestDescriptorComputation() {
    cv::Mat image(10, 10, CV_8UC1, cv::Scalar(128));
    std::vector<cv::Point> corners = {cv::Point(5, 5)};

    cv::Mat descriptors;
    ComputeSimpleDescriptors(image, corners, descriptors, 3);

    if (descriptors.rows == 1 && descriptors.cols == 9) {
        testsPassed++;
        std::cout << "[PASSED] TestDescriptorComputation: shape 1x9" << std::endl;
    } else {
        testsFailed++;
        std::cerr << "[FAIL] TestDescriptorComputation: shape "
                  << descriptors.rows << "x" << descriptors.cols << std::endl;
    }

    // All pixels are 128, so all descriptor values should be 128
    bool allEqual = true;
    for (int i = 0; i < descriptors.cols; i++) {
        if (std::abs(descriptors.at<float>(0, i) - 128.0f) > 0.01f) {
            allEqual = false;
            break;
        }
    }
    if (allEqual) {
        testsPassed++;
        std::cout << "[PASSED] TestDescriptorComputation: uniform values" << std::endl;
    } else {
        testsFailed++;
        std::cerr << "[FAIL] TestDescriptorComputation: values not uniform" << std::endl;
    }
}

void TestEuclideanDistance() {
    DescriptorMatcher matcher;

    cv::Mat desc1 = (cv::Mat_<float>(1, 3) << 0.0f, 0.0f, 0.0f);
    cv::Mat desc2 = (cv::Mat_<float>(1, 3) << 3.0f, 4.0f, 0.0f);

    float dist = matcher.ComputeEuclideanDistance(desc1, desc2);
    if (std::abs(dist - 5.0f) < 0.01f) {
        testsPassed++;
        std::cout << "[PASSED] TestEuclideanDistance: 5.0" << std::endl;
    } else {
        testsFailed++;
        std::cerr << "[FAIL] TestEuclideanDistance: expected 5.0, got " << dist << std::endl;
    }
}

void TestMatcherSelfMatch() {
    cv::Mat desc = (cv::Mat_<float>(2, 3) << 1.0f, 0.0f, 0.0f,
                                              0.0f, 1.0f, 0.0f);

    DescriptorMatcher matcher;
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point> corners = {cv::Point(0, 0), cv::Point(1, 1)};

    matcher.Match(desc, desc, matches, corners, corners);

    // Self-match should give perfect matches (distance 0, idx match)
    if (matches.size() == 2 && matches[0].queryIdx == 0 && matches[0].trainIdx == 0) {
        testsPassed++;
        std::cout << "[PASSED] TestMatcherSelfMatch" << std::endl;
    } else {
        testsFailed++;
        std::cerr << "[FAIL] TestMatcherSelfMatch" << std::endl;
    }
}

int main() {
    TestHarrisCornersOnCheckerboard();
    TestDescriptorComputation();
    TestEuclideanDistance();
    TestMatcherSelfMatch();

    std::cout << "\n" << testsPassed << " assertions passed, "
              << testsFailed << " failed." << std::endl;
    return testsFailed == 0 ? 0 : 1;
}
