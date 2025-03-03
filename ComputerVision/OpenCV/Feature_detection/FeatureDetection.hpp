/**
 * @file FeatureDetection.hpp
 * @brief ORB feature detection and BFMatcher-based matching
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>

namespace FeatureDetection {

struct MatchResult {
    std::vector<cv::KeyPoint> Keypoints1;
    std::vector<cv::KeyPoint> Keypoints2;
    cv::Mat Descriptors1;
    cv::Mat Descriptors2;
    std::vector<cv::DMatch> GoodMatches;
};

MatchResult DetectAndMatch(const cv::Mat& img1, const cv::Mat& img2,
                           int numGoodMatches = 30);

cv::Mat DrawMatchVisualization(const cv::Mat& img1, const cv::Mat& img2,
                               const MatchResult& result);

}  // namespace FeatureDetection
