#include "ImageProcessing.hpp"
namespace ImageProcessing {
cv::Mat ConvertToGray(const cv::Mat& bgr) {
    cv::Mat g; cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY); return g;
}
cv::Mat Resize(const cv::Mat& src, int w, int h) {
    cv::Mat dst; cv::resize(src, dst, cv::Size(w, h)); return dst;
}
cv::Mat GaussianBlur(const cv::Mat& src, int ks) {
    cv::Mat dst; cv::GaussianBlur(src, dst, cv::Size(ks, ks), 0); return dst;
}
cv::Mat Threshold(const cv::Mat& gray, double t, double mv) {
    cv::Mat dst; cv::threshold(gray, dst, t, mv, cv::THRESH_BINARY); return dst;
}
cv::Mat CannyEdge(const cv::Mat& gray, double lo, double hi) {
    cv::Mat dst; cv::Canny(gray, dst, lo, hi); return dst;
}
cv::Mat Rotate(const cv::Mat& src, double angle, double scale) {
    cv::Point2f center(src.cols/2.0f, src.rows/2.0f);
    cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);
    cv::Mat dst; cv::warpAffine(src, dst, M, src.size()); return dst;
}
}
