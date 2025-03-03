#include "SobelEdgeDetection.hpp"
#include <cmath>
#include <algorithm>

namespace SobelEdgeDetection {

SobelResult DetectEdges(const cv::Mat& bgrImage, double threshold) {
    SobelResult result;
    cv::Mat gray;
    cv::cvtColor(bgrImage, gray, cv::COLOR_BGR2GRAY);

    cv::Mat sobelX, sobelY;
    cv::Sobel(gray, sobelX, -1, 1, 0, 3);
    cv::Sobel(gray, sobelY, -1, 0, 1, 3);

    cv::Mat absX, absY;
    cv::convertScaleAbs(sobelX, absX);
    cv::convertScaleAbs(sobelY, absY);

    cv::addWeighted(absX, 0.5, absY, 0.5, 0, result.Combined);
    cv::threshold(result.Combined, result.Thresholded, threshold, 255, cv::THRESH_BINARY);

    result.GradientX = absX;
    result.GradientY = absY;
    return result;
}

cv::Mat CustomCvtColor(const cv::Mat& bgr) {
    cv::Mat dst(bgr.rows, bgr.cols, CV_8UC1);
    for (int y = 0; y < bgr.rows; ++y)
        for (int x = 0; x < bgr.cols; ++x) {
            auto px = bgr.at<cv::Vec3b>(y, x);
            dst.at<uchar>(y, x) = static_cast<uchar>(0.114*px[0] + 0.587*px[1] + 0.299*px[2]);
        }
    return dst;
}

void CustomSobel(const cv::Mat& gray, cv::Mat& gradX, cv::Mat& gradY) {
    gradX = cv::Mat::zeros(gray.size(), CV_16S);
    gradY = cv::Mat::zeros(gray.size(), CV_16S);
    for (int y = 1; y < gray.rows - 1; ++y)
        for (int x = 1; x < gray.cols - 1; ++x) {
            int gx = -gray.at<uchar>(y-1,x-1) + gray.at<uchar>(y-1,x+1)
                     -2*gray.at<uchar>(y,x-1) + 2*gray.at<uchar>(y,x+1)
                     -gray.at<uchar>(y+1,x-1) + gray.at<uchar>(y+1,x+1);
            int gy = -gray.at<uchar>(y-1,x-1) - 2*gray.at<uchar>(y-1,x) - gray.at<uchar>(y-1,x+1)
                     +gray.at<uchar>(y+1,x-1) + 2*gray.at<uchar>(y+1,x) + gray.at<uchar>(y+1,x+1);
            gradX.at<short>(y, x) = static_cast<short>(gx);
            gradY.at<short>(y, x) = static_cast<short>(gy);
        }
}

cv::Mat CustomConvertScaleAbs(const cv::Mat& src) {
    cv::Mat dst(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            dst.at<uchar>(y, x) = static_cast<uchar>(std::min(255, std::abs(src.at<short>(y, x))));
    return dst;
}

cv::Mat CustomAddWeighted(const cv::Mat& s1, double a, const cv::Mat& s2, double b, double g) {
    cv::Mat dst(s1.size(), CV_8UC1);
    for (int y = 0; y < s1.rows; ++y)
        for (int x = 0; x < s1.cols; ++x) {
            double v = a*s1.at<uchar>(y,x) + b*s2.at<uchar>(y,x) + g;
            dst.at<uchar>(y, x) = static_cast<uchar>(std::clamp(v, 0.0, 255.0));
        }
    return dst;
}

cv::Mat CustomThreshold(const cv::Mat& src, double thresh, double maxVal) {
    cv::Mat dst(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            dst.at<uchar>(y, x) = (src.at<uchar>(y, x) > thresh) ? static_cast<uchar>(maxVal) : 0;
    return dst;
}

}  // namespace SobelEdgeDetection
