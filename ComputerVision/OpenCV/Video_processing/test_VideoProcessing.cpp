#include "VideoProcessing.hpp"
#include <cassert>
#include <iostream>
int main() {
    std::cout << "Running VideoProcessing tests...\n";
    cv::Mat bgr(100, 100, CV_8UC3, cv::Scalar(50, 100, 150));
    auto result = VideoProcessing::ConvertToGrayBgr(bgr);
    assert(result.channels() == 3);
    assert(result.size() == bgr.size());
    auto px = result.at<cv::Vec3b>(0, 0);
    assert(px[0] == px[1] && px[1] == px[2]);
    std::cout << "All VideoProcessing tests PASSED.\n";
    return 0;
}
