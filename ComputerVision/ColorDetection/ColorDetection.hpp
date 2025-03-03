/**
 * @file ColorDetection.hpp
 * @brief HSV-based color detection with morphological filtering
 * @details Detects objects of a specified color range in HSV space. Includes
 *          custom implementations of inRange, morphologyEx (OPEN), contour
 *          finding, and bounding-rect computation.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace ColorDetection {

/// HSV range for a target color.
struct HsvRange {
    cv::Scalar Lower;
    cv::Scalar Upper;
};

/// Result of a single detected object.
struct DetectedObject {
    cv::Rect   BoundingBox;
    int        ContourArea;
};

// ---------------------------------------------------------------------------
// OpenCV-wrapper API (uses cv:: functions internally)
// ---------------------------------------------------------------------------

/**
 * @brief Convert a BGR image to HSV.
 * @param bgrImage  Input 3-channel BGR image.
 * @return HSV image.
 */
cv::Mat ConvertToHsv(const cv::Mat& bgrImage);

/**
 * @brief Create a binary mask of pixels inside the given HSV range.
 */
cv::Mat CreateColorMask(const cv::Mat& hsvImage, const HsvRange& range);

/**
 * @brief Clean a binary mask with morphological opening.
 * @param mask       Input/output single-channel mask.
 * @param kernelSize Side length of the rectangular structuring element.
 */
void CleanMask(cv::Mat& mask, int kernelSize = 5);

/**
 * @brief Detect objects whose color falls in the given HSV range.
 * @param bgrImage     Input 3-channel BGR image.
 * @param range        HSV range to search.
 * @param minSize      Minimum bounding-box side length to keep.
 * @param kernelSize   Structuring element size for morphological opening.
 * @return Vector of DetectedObject results.
 */
std::vector<DetectedObject> Detect(const cv::Mat& bgrImage,
                                   const HsvRange& range,
                                   int minSize = 20,
                                   int kernelSize = 5);

// ---------------------------------------------------------------------------
// Custom (manual) implementations -- educational, no cv:: algorithms inside
// ---------------------------------------------------------------------------

/**
 * @brief Manual inRange: creates a binary mask for pixels within HSV bounds.
 */
cv::Mat CustomInRange(const cv::Mat& hsvImage,
                      const cv::Scalar& lower,
                      const cv::Scalar& upper);

/**
 * @brief Manual rectangular structuring element (all ones).
 */
cv::Mat CustomStructuringElement(cv::Size size);

/**
 * @brief Manual morphological OPEN (erode then dilate).
 */
void CustomMorphOpen(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel);

/**
 * @brief Manual flood-fill contour finder (4-connected, external only).
 * @param binaryImage  Single-channel binary image (0 or 255).
 * @return Vector of contour point-sets, each with > minPoints points.
 */
std::vector<std::vector<cv::Point>> CustomFindContours(const cv::Mat& binaryImage,
                                                       int minPoints = 10);

/**
 * @brief Manual bounding rectangle from a point set.
 */
cv::Rect CustomBoundingRect(const std::vector<cv::Point>& contour);

}  // namespace ColorDetection
