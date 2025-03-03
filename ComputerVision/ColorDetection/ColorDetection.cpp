/**
 * @file ColorDetection.cpp
 * @brief Implementation of HSV-based color detection
 */

#include "ColorDetection.hpp"
#include <algorithm>

namespace ColorDetection {

// ---------- OpenCV-wrapper helpers ----------

cv::Mat ConvertToHsv(const cv::Mat& bgrImage) {
    cv::Mat hsv;
    cv::cvtColor(bgrImage, hsv, cv::COLOR_BGR2HSV);
    return hsv;
}

cv::Mat CreateColorMask(const cv::Mat& hsvImage, const HsvRange& range) {
    cv::Mat mask;
    cv::inRange(hsvImage, range.Lower, range.Upper, mask);
    return mask;
}

void CleanMask(cv::Mat& mask, int kernelSize) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
}

std::vector<DetectedObject> Detect(const cv::Mat& bgrImage,
                                   const HsvRange& range,
                                   int minSize,
                                   int kernelSize) {
    cv::Mat hsv  = ConvertToHsv(bgrImage);
    cv::Mat mask = CreateColorMask(hsv, range);
    CleanMask(mask, kernelSize);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<DetectedObject> results;
    for (const auto& contour : contours) {
        cv::Rect box = cv::boundingRect(contour);
        if (box.width >= minSize && box.height >= minSize) {
            results.push_back({box, static_cast<int>(cv::contourArea(contour))});
        }
    }
    return results;
}

// ---------- Custom implementations ----------

cv::Mat CustomInRange(const cv::Mat& hsvImage,
                      const cv::Scalar& lower,
                      const cv::Scalar& upper) {
    cv::Mat dst(hsvImage.size(), CV_8UC1);
    for (int y = 0; y < hsvImage.rows; ++y) {
        for (int x = 0; x < hsvImage.cols; ++x) {
            cv::Vec3b px = hsvImage.at<cv::Vec3b>(y, x);
            bool inside = (px[0] >= lower[0] && px[0] <= upper[0] &&
                           px[1] >= lower[1] && px[1] <= upper[1] &&
                           px[2] >= lower[2] && px[2] <= upper[2]);
            dst.at<uchar>(y, x) = inside ? 255 : 0;
        }
    }
    return dst;
}

cv::Mat CustomStructuringElement(cv::Size size) {
    return cv::Mat(size, CV_8UC1, cv::Scalar(1));
}

void CustomMorphOpen(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel) {
    int halfX = kernel.cols / 2;
    int halfY = kernel.rows / 2;

    // Erode
    cv::Mat eroded = src.clone();
    for (int y = halfY; y < src.rows - halfY; ++y) {
        for (int x = halfX; x < src.cols - halfX; ++x) {
            uchar minVal = 255;
            for (int ky = -halfY; ky <= halfY; ++ky)
                for (int kx = -halfX; kx <= halfX; ++kx)
                    if (kernel.at<uchar>(ky + halfY, kx + halfX))
                        minVal = std::min(minVal, src.at<uchar>(y + ky, x + kx));
            eroded.at<uchar>(y, x) = minVal;
        }
    }

    // Dilate
    dst = eroded.clone();
    for (int y = halfY; y < eroded.rows - halfY; ++y) {
        for (int x = halfX; x < eroded.cols - halfX; ++x) {
            uchar maxVal = 0;
            for (int ky = -halfY; ky <= halfY; ++ky)
                for (int kx = -halfX; kx <= halfX; ++kx)
                    if (kernel.at<uchar>(ky + halfY, kx + halfX))
                        maxVal = std::max(maxVal, eroded.at<uchar>(y + ky, x + kx));
            dst.at<uchar>(y, x) = maxVal;
        }
    }
}

std::vector<std::vector<cv::Point>> CustomFindContours(const cv::Mat& binaryImage,
                                                       int minPoints) {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat visited = cv::Mat::zeros(binaryImage.size(), CV_8UC1);

    for (int y = 0; y < binaryImage.rows; ++y) {
        for (int x = 0; x < binaryImage.cols; ++x) {
            if (binaryImage.at<uchar>(y, x) == 255 && !visited.at<uchar>(y, x)) {
                std::vector<cv::Point> contour;
                std::vector<cv::Point> stack;
                stack.emplace_back(x, y);

                while (!stack.empty()) {
                    cv::Point p = stack.back();
                    stack.pop_back();
                    if (visited.at<uchar>(p.y, p.x)) continue;
                    visited.at<uchar>(p.y, p.x) = 255;
                    contour.push_back(p);

                    constexpr int DIRS[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};
                    for (auto& d : DIRS) {
                        int ny = p.y + d[0], nx = p.x + d[1];
                        if (ny >= 0 && ny < binaryImage.rows &&
                            nx >= 0 && nx < binaryImage.cols &&
                            binaryImage.at<uchar>(ny, nx) == 255 &&
                            !visited.at<uchar>(ny, nx)) {
                            stack.emplace_back(nx, ny);
                        }
                    }
                }
                if (static_cast<int>(contour.size()) > minPoints)
                    contours.push_back(std::move(contour));
            }
        }
    }
    return contours;
}

cv::Rect CustomBoundingRect(const std::vector<cv::Point>& contour) {
    int minX = contour[0].x, maxX = contour[0].x;
    int minY = contour[0].y, maxY = contour[0].y;
    for (const auto& p : contour) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }
    return cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
}

}  // namespace ColorDetection
