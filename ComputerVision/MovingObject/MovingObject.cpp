/**
 * @file MovingObject.cpp
 * @brief Implementation of ORB-based motion detection
 */

#include "MovingObject.hpp"
#include <sstream>

namespace MovingObject {

MotionDetector::MotionDetector(int numFeatures, float matchingThreshold, int minInliers)
    : numFeatures_(numFeatures),
      matchingThreshold_(matchingThreshold),
      minInliers_(minInliers) {}

void MotionDetector::UpdateParameters(int nf, float mt, int mi) {
    std::lock_guard<std::mutex> lock(detectorMutex_);
    numFeatures_      = nf;
    matchingThreshold_ = mt;
    minInliers_       = mi;
}

std::vector<std::pair<float, cv::Point2f>>
MotionDetector::CalculateVelocities(const std::vector<cv::Point2f>& points1,
                                     const std::vector<cv::Point2f>& points2,
                                     float timeInterval) {
    if (timeInterval <= 0)
        throw std::invalid_argument("Time interval must be positive");
    if (points1.size() != points2.size())
        throw std::invalid_argument("Point arrays must match in size");

    std::vector<std::pair<float, cv::Point2f>> velocities;
    velocities.reserve(points1.size());

    for (size_t i = 0; i < points1.size(); ++i) {
        cv::Point2f vel = (points2[i] - points1[i]) * (1.0f / timeInterval);
        float mag = static_cast<float>(cv::norm(vel));
        velocities.emplace_back(mag, vel);
    }
    return velocities;
}

cv::Mat MotionDetector::DetectMotion(const cv::Mat& img1, const cv::Mat& img2,
                                      cv::Mat& motionMask, float timeInterval) {
    if (img1.empty() || img2.empty())
        throw std::invalid_argument("Input images cannot be empty");
    if (img1.size() != img2.size())
        throw std::invalid_argument("Images must have the same dimensions");
    if (timeInterval <= 0)
        throw std::invalid_argument("Time interval must be positive");

    std::lock_guard<std::mutex> lock(detectorMutex_);

    cv::Mat gray1, gray2;
    if (img1.channels() == 3) {
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    } else {
        gray1 = img1.clone();
        gray2 = img2.clone();
    }

    auto orb = cv::ORB::create(numFeatures_);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    orb->detectAndCompute(gray1, cv::noArray(), kp1, desc1);
    orb->detectAndCompute(gray2, cv::noArray(), kp2, desc2);

    if (kp1.empty() || kp2.empty() || desc1.empty() || desc2.empty()) {
        motionMask = cv::Mat::zeros(img1.size(), CV_8UC1);
        return img2.clone();
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    std::vector<cv::DMatch> goodMatches;
    for (const auto& m : knnMatches) {
        if (m.size() >= 2 && m[0].distance < matchingThreshold_ * m[1].distance)
            goodMatches.push_back(m[0]);
    }

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : goodMatches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    auto velocities = CalculateVelocities(pts1, pts2, timeInterval);
    motionMask = cv::Mat::zeros(img1.size(), CV_8UC1);
    cv::Mat result = img2.clone();

    if (goodMatches.size() >= 4) {
        std::vector<uchar> inliersMask;
        cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, inliersMask);

        if (!H.empty() && cv::countNonZero(inliersMask) >= minInliers_) {
            cv::Mat warped;
            cv::warpPerspective(img1, warped, H, img1.size());

            cv::Mat diff;
            cv::absdiff(warped, img2, diff);

            cv::Mat diffGray;
            if (diff.channels() == 3)
                cv::cvtColor(diff, diffGray, cv::COLOR_BGR2GRAY);
            else
                diffGray = diff;

            cv::threshold(diffGray, motionMask, 30, 255, cv::THRESH_BINARY);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::morphologyEx(motionMask, motionMask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, kernel);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(motionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(result, contours, -1, cv::Scalar(0, 0, 255), 2);

            for (const auto& contour : contours) {
                if (cv::contourArea(contour) < 100) continue;
                cv::Rect bbox = cv::boundingRect(contour);
                cv::rectangle(result, bbox, cv::Scalar(0, 255, 0), 2);

                float totalVel = 0;
                int   count    = 0;
                cv::Point2f avgDir(0, 0);
                for (size_t i = 0; i < goodMatches.size(); ++i) {
                    if (bbox.contains(pts2[i])) {
                        totalVel += velocities[i].first;
                        avgDir   += velocities[i].second;
                        ++count;
                    }
                }
                if (count > 0) {
                    float avgVel = totalVel / count;
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(1) << avgVel << " px/s";
                    cv::putText(result, ss.str(),
                                cv::Point(bbox.x, bbox.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(0, 255, 0), 2);
                }
            }
        }
    }
    return result;
}

}  // namespace MovingObject
