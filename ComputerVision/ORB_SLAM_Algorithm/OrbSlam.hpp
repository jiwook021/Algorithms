/**
 * @file OrbSlam.hpp
 * @brief Harris corner detection with simple patch descriptors
 * @details Detects corners using Harris response, extracts 5x5 patch
 *          descriptors around each keypoint.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace OrbSlam {

/**
 * @brief Detect Harris corners in a grayscale image.
 * @param image      Single-channel input.
 * @param corners    Output corner points.
 * @param blockSize  Neighbourhood size for corner detection.
 * @param kSize      Sobel operator aperture.
 * @param k          Harris free parameter.
 */
void DetectHarrisCorners(const cv::Mat& image,
                         std::vector<cv::Point>& corners,
                         int blockSize = 2,
                         int kSize     = 3,
                         double k      = 0.04);

/**
 * @brief Extract simple NxN patch descriptors around each corner.
 * @param image       Single-channel input.
 * @param corners     Detected corner points.
 * @param descriptors Output descriptor matrix (one row per corner).
 * @param patchSize   Side length of the patch.
 */
void ComputeSimpleDescriptors(const cv::Mat& image,
                               const std::vector<cv::Point>& corners,
                               cv::Mat& descriptors,
                               int patchSize = 5);

}  // namespace OrbSlam
