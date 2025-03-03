#include "ImageProcessing.hpp"
#include <iostream>
int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "input.jpg";
    cv::Mat img = cv::imread(path);
    if (img.empty()) { std::cerr << "Cannot load image\n"; return -1; }
    auto gray = ImageProcessing::ConvertToGray(img);
    auto edges = ImageProcessing::CannyEdge(gray);
    cv::imshow("Original", img);
    cv::imshow("Edges", edges);
    cv::waitKey(0);
    return 0;
}
