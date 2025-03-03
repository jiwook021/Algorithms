#include "ImageSegmentation.hpp"
#include <iostream>
int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "input.jpg";
    cv::Mat img = cv::imread(path);
    if (img.empty()) { std::cerr << "Cannot load\n"; return -1; }
    auto binary = ImageSegmentation::Binarize(img);
    auto objects = ImageSegmentation::FindObjects(binary);
    auto vis = ImageSegmentation::DrawSegmentation(img, objects);
    std::cout << "Objects: " << objects.size() << "\n";
    cv::imshow("Segmentation", vis);
    cv::waitKey(0);
    return 0;
}
