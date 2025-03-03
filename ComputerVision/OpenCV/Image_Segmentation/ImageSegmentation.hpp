/**
 * @file ImageSegmentation.hpp
 * @brief Contour-based image segmentation with Otsu thresholding
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
namespace ImageSegmentation {
struct SegmentedObject {
    std::vector<cv::Point> Contour;
    cv::Rect BoundingBox;
    cv::RotatedRect RotatedBox;
    cv::Point2f Centroid;
    double Area;
};
cv::Mat Binarize(const cv::Mat& bgr);
std::vector<SegmentedObject> FindObjects(const cv::Mat& binary, double minArea = 500);
cv::Mat DrawSegmentation(const cv::Mat& bgr, const std::vector<SegmentedObject>& objects);
}
