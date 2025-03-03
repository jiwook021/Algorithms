#include "ImageSegmentation.hpp"
namespace ImageSegmentation {
cv::Mat Binarize(const cv::Mat& bgr) {
    cv::Mat gray, blurred, binary;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5,5), 0);
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    return binary;
}
std::vector<SegmentedObject> FindObjects(const cv::Mat& binary, double minArea) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<SegmentedObject> objects;
    for (auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < minArea) continue;
        SegmentedObject obj;
        obj.Contour = c;
        obj.BoundingBox = cv::boundingRect(c);
        obj.RotatedBox = cv::minAreaRect(c);
        obj.Area = area;
        cv::Moments m = cv::moments(c);
        obj.Centroid = (m.m00 != 0) ? cv::Point2f(m.m10/m.m00, m.m01/m.m00) : cv::Point2f(0,0);
        objects.push_back(obj);
    }
    return objects;
}
cv::Mat DrawSegmentation(const cv::Mat& bgr, const std::vector<SegmentedObject>& objects) {
    cv::Mat drawing = bgr.clone();
    for (size_t i = 0; i < objects.size(); ++i) {
        cv::drawContours(drawing, std::vector<std::vector<cv::Point>>{objects[i].Contour},
                         0, cv::Scalar(0,255,0), 2);
        cv::rectangle(drawing, objects[i].BoundingBox, cv::Scalar(255,0,0), 2);
        cv::circle(drawing, objects[i].Centroid, 5, cv::Scalar(255,0,255), -1);
    }
    return drawing;
}
}
