/**
 * @file FastObjectDetection.hpp
 * @brief Fast Moving Object (FMO) detection using background subtraction
 * @details Detects elongated, high-speed objects via MOG2 background subtraction,
 *          contour analysis, and optical flow motion estimation.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

namespace FastObjectDetection {

/// Detected fast-moving object.
struct FmoObject {
    cv::Rect     BoundingBox;
    cv::Point2f  Direction;
    float        Speed;
    cv::Mat      Appearance;
};

class FmoDetector {
public:
    /**
     * @param thresholdDiff  Background difference threshold.
     * @param minArea        Minimum contour area.
     * @param maxAspectRatio Maximum aspect ratio for FMO candidates.
     * @param historySize    Background model history length.
     */
    FmoDetector(int thresholdDiff  = 30,
                int minArea        = 50,
                float maxAspectRatio = 5.0f,
                int historySize    = 20);

    /// Detect fast-moving objects in a single frame.
    std::vector<FmoObject> Detect(const cv::Mat& frame);

private:
    int   thresholdDiff_;
    int   minArea_;
    float maxAspectRatio_;
    int   historySize_;

    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor_;
    std::vector<cv::Mat> frameHistory_;

    void UpdateFrameHistory(const cv::Mat& grayFrame);

    void ComputeMotionFeatures(const cv::Mat& currentFrame,
                               const cv::Rect& bbox,
                               cv::Point2f& direction,
                               float& speed);

    static cv::Rect ExpandRect(const cv::Rect& rect,
                               const cv::Size& imgSize,
                               float scale);
};

}  // namespace FastObjectDetection
