/**
 * @file MovingObject.hpp
 * @brief ORB-based motion detection with velocity estimation
 * @details Detects moving objects between two images using ORB feature matching,
 *          homography-based background alignment, and velocity estimation.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <mutex>
#include <string>
#include <iomanip>

namespace MovingObject {

class MotionDetector {
public:
    /**
     * @param numFeatures      Number of ORB features to extract.
     * @param matchingThreshold Lowe's ratio test threshold.
     * @param minInliers       Minimum homography inliers.
     */
    MotionDetector(int numFeatures      = 500,
                   float matchingThreshold = 0.75f,
                   int minInliers       = 10);

    /**
     * @brief Detect motion between two frames.
     * @param img1          Previous frame.
     * @param img2          Current frame.
     * @param motionMask    Output binary mask of moving regions.
     * @param timeInterval  Seconds between frames (for velocity calculation).
     * @return Annotated visualisation image.
     */
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

}  // namespace MovingObject
