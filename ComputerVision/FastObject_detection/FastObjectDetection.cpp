/**
 * @file FastObjectDetection.cpp
 * @brief Implementation of FMO detection
 */

#include "FastObjectDetection.hpp"

namespace FastObjectDetection {

FmoDetector::FmoDetector(int thresholdDiff, int minArea,
                         float maxAspectRatio, int historySize)
    : thresholdDiff_(thresholdDiff),
      minArea_(minArea),
      maxAspectRatio_(maxAspectRatio),
      historySize_(historySize) {
    bgSubtractor_ = cv::createBackgroundSubtractorMOG2(historySize, 16, false);
}

std::vector<FmoObject> FmoDetector::Detect(const cv::Mat& frame) {
    std::vector<FmoObject> detectedFmos;
    if (frame.empty()) return detectedFmos;

    cv::Mat grayFrame;
    if (frame.channels() == 3)
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    else
        grayFrame = frame.clone();

    UpdateFrameHistory(grayFrame);

    cv::Mat foregroundMask;
    bgSubtractor_->apply(grayFrame, foregroundMask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(foregroundMask, foregroundMask, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(foregroundMask, foregroundMask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(foregroundMask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < minArea_) continue;

        cv::Rect bbox = cv::boundingRect(contour);
        float aspectRatio = static_cast<float>(bbox.width) / bbox.height;

        if (aspectRatio > maxAspectRatio_ ||
            aspectRatio < 1.0f / maxAspectRatio_) {
            cv::Point2f direction;
            float speed;
            ComputeMotionFeatures(grayFrame, bbox, direction, speed);

            if (speed > 5.0f) {
                FmoObject fmo;
                fmo.BoundingBox = bbox;
                fmo.Direction   = direction;
                fmo.Speed       = speed;
                fmo.Appearance  = frame(bbox).clone();
                detectedFmos.push_back(fmo);
            }
        }
    }
    return detectedFmos;
}

void FmoDetector::UpdateFrameHistory(const cv::Mat& grayFrame) {
    frameHistory_.push_back(grayFrame.clone());
    if (static_cast<int>(frameHistory_.size()) > historySize_)
        frameHistory_.erase(frameHistory_.begin());
}

void FmoDetector::ComputeMotionFeatures(const cv::Mat& currentFrame,
                                         const cv::Rect& bbox,
                                         cv::Point2f& direction,
                                         float& speed) {
    direction = cv::Point2f(0, 0);
    speed     = 0.0f;

    if (frameHistory_.size() < 2) return;

    cv::Rect searchArea = ExpandRect(bbox, currentFrame.size(), 1.5f);
    cv::Mat prevFrame = frameHistory_[frameHistory_.size() - 2];

    std::vector<cv::Point2f> currPoints;
    cv::Mat roi = currentFrame(searchArea);
    cv::goodFeaturesToTrack(roi, currPoints, 100, 0.01, 10);

    for (auto& pt : currPoints) {
        pt.x += searchArea.x;
        pt.y += searchArea.y;
    }
    if (currPoints.empty()) return;

    std::vector<cv::Point2f> prevPoints;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevFrame, currentFrame, currPoints,
                             prevPoints, status, err);

    cv::Point2f avgMotion(0, 0);
    int validPoints = 0;
    for (size_t i = 0; i < currPoints.size(); ++i) {
        if (status[i]) {
            avgMotion += currPoints[i] - prevPoints[i];
            ++validPoints;
        }
    }

    if (validPoints > 0) {
        avgMotion /= static_cast<float>(validPoints);
        direction = avgMotion;
        speed     = static_cast<float>(cv::norm(avgMotion));
    }
}

cv::Rect FmoDetector::ExpandRect(const cv::Rect& rect,
                                  const cv::Size& imgSize,
                                  float scale) {
    int widthDiff  = static_cast<int>(rect.width  * (scale - 1));
    int heightDiff = static_cast<int>(rect.height * (scale - 1));

    cv::Rect expanded;
    expanded.x      = std::max(0, rect.x - widthDiff / 2);
    expanded.y      = std::max(0, rect.y - heightDiff / 2);
    expanded.width  = std::min(imgSize.width  - expanded.x, rect.width  + widthDiff);
    expanded.height = std::min(imgSize.height - expanded.y, rect.height + heightDiff);
    return expanded;
}

}  // namespace FastObjectDetection
