/**
 * @file OrbSlam.cpp
 * @brief Implementation of Harris corner detection and patch descriptors
 */

#include "OrbSlam.hpp"

namespace OrbSlam {

void DetectHarrisCorners(const cv::Mat& image,
                         std::vector<cv::Point>& corners,
                         int blockSize, int kSize, double k) {
    cv::Mat dst = cv::Mat::zeros(image.size(), CV_32FC1);
    cv::cornerHarris(image, dst, blockSize, kSize, k);

    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal);
    double threshold = 0.01 * maxVal;

    for (int y = 0; y < dst.rows; ++y)
        for (int x = 0; x < dst.cols; ++x)
            if (dst.at<float>(y, x) > threshold)
                corners.emplace_back(x, y);
}

void ComputeSimpleDescriptors(const cv::Mat& image,
                               const std::vector<cv::Point>& corners,
                               cv::Mat& descriptors,
                               int patchSize) {
    int halfPatch = patchSize / 2;
    descriptors = cv::Mat::zeros(static_cast<int>(corners.size()),
                                  patchSize * patchSize, CV_32FC1);

    for (size_t i = 0; i < corners.size(); ++i) {
        int cx = corners[i].x, cy = corners[i].y;
        int idx = 0;
        for (int dy = -halfPatch; dy <= halfPatch; ++dy) {
            for (int dx = -halfPatch; dx <= halfPatch; ++dx) {
                int px = cx + dx, py = cy + dy;
                if (px >= 0 && px < image.cols && py >= 0 && py < image.rows)
                    descriptors.at<float>(static_cast<int>(i), idx) =
                        static_cast<float>(image.at<uchar>(py, px));
                ++idx;
            }
        }
    }
}

}  // namespace OrbSlam
