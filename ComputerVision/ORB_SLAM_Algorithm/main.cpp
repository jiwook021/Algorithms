/**
 * @file main.cpp
 * @brief Driver for OrbSlam module
 */

#include "OrbSlam.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string imagePath = (argc > 1) ? argv[1] : "1.png";

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image: " << imagePath << std::endl;
        return 1;
    }

    std::vector<cv::Point> corners;
    OrbSlam::DetectHarrisCorners(image, corners);

    cv::Mat descriptors;
    OrbSlam::ComputeSimpleDescriptors(image, corners, descriptors);

    std::cout << "Corners detected: " << corners.size() << "\n"
              << "Descriptor size: " << descriptors.cols << "\n";

    cv::Mat output = image.clone();
    for (const auto& corner : corners)
        cv::circle(output, corner, 5, cv::Scalar(255), 2);

    cv::imshow("Detected Corners", output);
    cv::waitKey(0);
    return 0;
}
