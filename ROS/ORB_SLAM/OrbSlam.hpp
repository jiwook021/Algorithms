#ifndef ORB_SLAM_HPP
#define ORB_SLAM_HPP

/**
 * @file OrbSlam.hpp
 * @brief Header for simplified ORB-SLAM feature matching pipeline.
 *
 * Implements Harris corner detection, patch-based descriptor computation,
 * brute-force Euclidean matching, and essential matrix recovery.
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>

/**
 * @brief Detect Harris corners in a grayscale image.
 *
 * @param image     Input grayscale image.
 * @param corners   Output vector of detected corner positions.
 * @param blockSize Harris block size (default: 2).
 * @param kSize     Sobel kernel size (default: 3).
 * @param k         Harris free parameter (default: 0.04).
 */
void DetectHarrisCorners(const cv::Mat& image, std::vector<cv::Point>& corners,
                         int blockSize = 2, int kSize = 3, double k = 0.04);

/**
 * @brief Compute simple patch-based descriptors around detected corners.
 *
 * @param image       Input grayscale image.
 * @param corners     Corner positions.
 * @param descriptors Output descriptor matrix (one row per corner).
 * @param patchSize   Size of the descriptor patch (default: 5).
 */
void ComputeSimpleDescriptors(const cv::Mat& image,
                               const std::vector<cv::Point>& corners,
                               cv::Mat& descriptors, int patchSize = 5);

/**
 * @class DescriptorMatcher
 * @brief Brute-force descriptor matcher using Euclidean distance.
 */
class DescriptorMatcher {
public:
    /**
     * @brief Compute Euclidean distance between two descriptor rows.
     */
    float ComputeEuclideanDistance(const cv::Mat& desc1, const cv::Mat& desc2);

    /**
     * @brief Match descriptors from image 1 to image 2 using brute-force nearest neighbor.
     */
    void Match(const cv::Mat& descriptors1, const cv::Mat& descriptors2,
               std::vector<cv::DMatch>& matches,
               const std::vector<cv::Point>& corners1,
               const std::vector<cv::Point>& corners2);
};

#endif // ORB_SLAM_HPP
