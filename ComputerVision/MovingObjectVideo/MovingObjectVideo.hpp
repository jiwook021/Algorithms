/**
 * @file MovingObjectVideo.hpp
 * @brief Video-based motion detection and tracking with velocity estimation
 * @details Extends ORB-based motion detection to video streams. Processes
 *          consecutive frame pairs with configurable frame skipping.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <mutex>
#include <string>
#include <iomanip>
#include <sstream>

namespace MovingObjectVideo {

class MotionDetector {
public:
    MotionDetector(int numFeatures      = 500,
                   float matchingThreshold = 0.75f,
                   int minInliers       = 10);

    cv::Mat DetectMotion(const cv::Mat& img1,
                         const cv::Mat& img2,
                         cv::Mat& motionMask,
                         float timeInterval = 0.5f);

    void UpdateParameters(int numFeatures, float matchingThreshold, int minInliers);

private:
    int   numFeatures_;
    float matchingThreshold_;
    int   minInliers_;
    std::mutex detectorMutex_;

    std::vector<std::pair<float, cv::Point2f>>
    CalculateVelocities(const std::vector<cv::Point2f>& points1,
                        const std::vector<cv::Point2f>& points2,
                        float timeInterval);
};

}  // namespace MovingObjectVideo
