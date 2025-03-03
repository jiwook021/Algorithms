#include "SobelEdgeDetection.hpp"
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: " << argv[0] << " <image>\n"; return -1; }
    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) { std::cerr << "Cannot load\n"; return -1; }
    auto result = SobelEdgeDetection::DetectEdges(img);
    cv::imshow("Sobel Edges", result.Combined);
    cv::imshow("Threshold", result.Thresholded);
    cv::waitKey(0);
    return 0;
}
