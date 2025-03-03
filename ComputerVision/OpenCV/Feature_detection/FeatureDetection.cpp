/**
 * @file FeatureDetection.cpp
 * @brief Implementation of ORB detection and matching
 */

#include "FeatureDetection.hpp"
#include <algorithm>

namespace FeatureDetection {

MatchResult DetectAndMatch(const cv::Mat& img1, const cv::Mat& img2,
                           int numGoodMatches) {
    MatchResult result;

    cv::Mat gray1, gray2;
    if (img1.channels() == 3) cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    else gray1 = img1;
    if (img2.channels() == 3) cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    else gray2 = img2;

    auto detector = cv::ORB::create();
    detector->detectAndCompute(gray1, cv::noArray(), result.Keypoints1, result.Descriptors1);
    detector->detectAndCompute(gray2, cv::noArray(), result.Keypoints2, result.Descriptors2);

    if (result.Descriptors1.empty() || result.Descriptors2.empty()) return result;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(result.Descriptors1, result.Descriptors2, matches);

    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

    int keep = std::min(numGoodMatches, static_cast<int>(matches.size()));
    result.GoodMatches.assign(matches.begin(), matches.begin() + keep);
    return result;
}

cv::Mat DrawMatchVisualization(const cv::Mat& img1, const cv::Mat& img2,
                               const MatchResult& result) {
    cv::Mat output;
    cv::drawMatches(img1, result.Keypoints1, img2, result.Keypoints2,
                    result.GoodMatches, output,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return output;
}

}  // namespace FeatureDetection
