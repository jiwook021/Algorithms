#include "ImageProcessing.hpp"
#include <cassert>
#include <iostream>
int main() {
    std::cout << "Running ImageProcessing tests...\n";
    cv::Mat bgr(100, 100, CV_8UC3, cv::Scalar(100, 150, 200));
    auto gray = ImageProcessing::ConvertToGray(bgr);
    assert(gray.channels() == 1);
    auto resized = ImageProcessing::Resize(bgr, 50, 50);
    assert(resized.cols == 50 && resized.rows == 50);
    auto blurred = ImageProcessing::GaussianBlur(bgr, 3);
    assert(blurred.size() == bgr.size());
    auto thresh = ImageProcessing::Threshold(gray);
    assert(thresh.size() == gray.size());
    auto edges = ImageProcessing::CannyEdge(gray);
    assert(edges.channels() == 1);
    auto rotated = ImageProcessing::Rotate(bgr, 45);
    assert(rotated.size() == bgr.size());
    std::cout << "All ImageProcessing tests PASSED.\n";
    return 0;
}
