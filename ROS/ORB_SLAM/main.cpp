/**
 * @file main.cpp
 * @brief Driver for ORB-SLAM feature matching and pose recovery.
 *
 * Loads two images, detects Harris corners, matches descriptors,
 * and recovers the essential matrix and camera pose.
 *
 * Dependencies: OpenCV
 */

#include "OrbSlam.hpp"
#include <iostream>

int main() {
    cv::Mat image1 = cv::imread("1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("2.png", cv::IMREAD_GRAYSCALE);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        return 1;
    }

    // Detect corners
    std::vector<cv::Point> corners1, corners2;
    DetectHarrisCorners(image1, corners1);
    DetectHarrisCorners(image2, corners2);
    std::cout << "Corners detected: " << corners1.size() << " / " << corners2.size() << std::endl;

    // Compute descriptors
    cv::Mat descriptors1, descriptors2;
    ComputeSimpleDescriptors(image1, corners1, descriptors1);
    ComputeSimpleDescriptors(image2, corners2, descriptors2);

    // Match descriptors
    DescriptorMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.Match(descriptors1, descriptors2, matches, corners1, corners2);
    std::cout << "Number of matches: " << matches.size() << std::endl;

    // Filter good matches
    double minDistance = 1e5, maxDistance = 0;
    for (const auto& match : matches) {
        if (match.distance > maxDistance) maxDistance = match.distance;
        if (match.distance < minDistance) minDistance = match.distance;
    }

    std::vector<cv::DMatch> goodMatches;
    for (const auto& match : matches) {
        if (match.distance <= std::max(minDistance * 1.6, 25.0)) {
            goodMatches.push_back(match);
        }
    }
    std::cout << "Good matches: " << goodMatches.size() << std::endl;

    // Recover pose from essential matrix
    cv::Point2d principalPoint(325.1, 249.7);
    double focalLength = 521;

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    for (const auto& c : corners1) keypoints1.push_back(cv::KeyPoint(c.x, c.y, 5.0));
    for (const auto& c : corners2) keypoints2.push_back(cv::KeyPoint(c.x, c.y, 5.0));

    if (goodMatches.size() >= 5) {
        std::vector<cv::Point2f> points1, points2;
        for (const auto& m : goodMatches) {
            points1.push_back(keypoints1[m.queryIdx].pt);
            points2.push_back(keypoints2[m.trainIdx].pt);
        }

        cv::Mat essentialMatrix = cv::findEssentialMat(points1, points2, focalLength, principalPoint);
        if (!essentialMatrix.empty()) {
            cv::Mat R, t;
            int inliers = cv::recoverPose(essentialMatrix, points1, points2, R, t, focalLength, principalPoint);
            std::cout << "Inliers from pose recovery: " << inliers << std::endl;
            std::cout << "Rotation Matrix (R):\n" << R << std::endl;
            std::cout << "Translation Vector (t):\n" << t << std::endl;
        }
    } else {
        std::cerr << "Not enough good matches for essential matrix (need >= 5)." << std::endl;
    }

    return 0;
}
