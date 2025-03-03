/**
 * @file main.cpp
 * @brief Driver for ColorDetection module
 */

#include "ColorDetection.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image" << std::endl;
        return -1;
    }

    // Detect red objects
    ColorDetection::HsvRange redRange{cv::Scalar(0, 120, 70),
                                       cv::Scalar(10, 255, 255)};
    auto results = ColorDetection::Detect(image, redRange);

    // Draw results
    cv::Mat display = image.clone();
    for (size_t i = 0; i < results.size(); ++i) {
        cv::rectangle(display, results[i].BoundingBox, cv::Scalar(0, 255, 0), 2);
        std::string label = "Object " + std::to_string(i);
        cv::putText(display, label, results[i].BoundingBox.tl(),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Detected Objects", display);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
