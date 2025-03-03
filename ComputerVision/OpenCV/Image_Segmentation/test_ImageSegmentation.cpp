#include "ImageSegmentation.hpp"
#include <cassert>
#include <iostream>
int main() {
    std::cout << "Running ImageSegmentation tests...\n";
    cv::Mat img(200, 200, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::rectangle(img, cv::Rect(50, 50, 60, 60), cv::Scalar(0, 0, 0), -1);
    auto bin = ImageSegmentation::Binarize(img);
    assert(bin.channels() == 1);
    auto objects = ImageSegmentation::FindObjects(bin, 100);
    assert(objects.size() >= 1);
    assert(objects[0].Area > 3000);
    auto vis = ImageSegmentation::DrawSegmentation(img, objects);
    assert(!vis.empty());
    std::cout << "All ImageSegmentation tests PASSED.\n";
    return 0;
}
