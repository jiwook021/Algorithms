/**
 * @file OrbSlam.cpp
 * @brief Implementation of simplified ORB-SLAM feature matching pipeline.
 *
 * Provides Harris corner detection, patch descriptors, and brute-force matching.
 */

#include "OrbSlam.hpp"

void DetectHarrisCorners(const cv::Mat& image, std::vector<cv::Point>& corners,
                         int blockSize, int kSize, double k) {
    cv::Mat dst = cv::Mat::zeros(image.size(), CV_32FC1);
    cv::cornerHarris(image, dst, blockSize, kSize, k);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc);

    double threshold = 0.01 * maxVal;
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            if (dst.at<float>(y, x) > threshold) {
                corners.push_back(cv::Point(x, y));
            }
        }
    }
}

void ComputeSimpleDescriptors(const cv::Mat& image,
                               const std::vector<cv::Point>& corners,
                               cv::Mat& descriptors, int patchSize) {
    int halfPatch = patchSize / 2;
    descriptors = cv::Mat::zeros(static_cast<int>(corners.size()),
                                  patchSize * patchSize, CV_32FC1);

    for (size_t i = 0; i < corners.size(); i++) {
        int x = corners[i].x;
        int y = corners[i].y;
        int idx = 0;
        for (int dy = -halfPatch; dy <= halfPatch; dy++) {
            for (int dx = -halfPatch; dx <= halfPatch; dx++) {
                int px = x + dx;
                int py = y + dy;
                if (px >= 0 && px < image.cols && py >= 0 && py < image.rows) {
                    descriptors.at<float>(i, idx) = static_cast<float>(image.at<uchar>(py, px));
                } else {
                    descriptors.at<float>(i, idx) = 0.0f;
                }
                idx++;
            }
        }
    }
}

float DescriptorMatcher::ComputeEuclideanDistance(const cv::Mat& desc1, const cv::Mat& desc2) {
    float distance = 0.0f;
    for (int i = 0; i < desc1.cols; ++i) {
        float diff = desc1.at<float>(0, i) - desc2.at<float>(0, i);
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

void DescriptorMatcher::Match(const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                               std::vector<cv::DMatch>& matches,
                               const std::vector<cv::Point>& corners1,
                               const std::vector<cv::Point>& corners2) {
    matches.clear();
    for (int i = 0; i < descriptors1.rows; ++i) {
        float minDist = std::numeric_limits<float>::max();
        int minIdx = -1;
        for (int j = 0; j < descriptors2.rows; ++j) {
            float dist = ComputeEuclideanDistance(descriptors1.row(i), descriptors2.row(j));
            if (dist < minDist) {
                minDist = dist;
                minIdx = j;
            }
        }
        if (minIdx != -1) {
            matches.push_back(cv::DMatch(i, minIdx, minDist));
        }
    }
}
