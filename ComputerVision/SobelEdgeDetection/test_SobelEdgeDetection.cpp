#include "SobelEdgeDetection.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

int main() {
    std::cout << "Running SobelEdgeDetection tests...\n";

    // Test on synthetic image with a sharp vertical edge
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    img(cv::Rect(50, 0, 50, 100)).setTo(cv::Scalar(255, 255, 255));

    auto result = SobelEdgeDetection::DetectEdges(img, 50);
    assert(!result.Combined.empty());
    assert(result.Thresholded.type() == CV_8UC1);
    // Edge should be detected near column 50
    assert(result.Combined.at<uchar>(50, 50) > 0);

    // Test custom grayscale
    cv::Mat gray = SobelEdgeDetection::CustomCvtColor(img);
    assert(gray.channels() == 1);
    assert(gray.at<uchar>(50, 0) == 0);
    assert(gray.at<uchar>(50, 99) == 255);

    // Test custom Sobel
    cv::Mat gx, gy;
    SobelEdgeDetection::CustomSobel(gray, gx, gy);
    assert(gx.at<short>(50, 50) != 0);  // edge produces gradient

    // Test custom threshold
    cv::Mat t = SobelEdgeDetection::CustomThreshold(gray, 127, 255);
    assert(t.at<uchar>(50, 0) == 0);
    assert(t.at<uchar>(50, 99) == 255);

    std::cout << "All SobelEdgeDetection tests PASSED.\n";
    return 0;
}
