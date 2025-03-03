/**
 * @file main.cpp
 * @brief Driver for MovingObject module
 */

#include "MovingObject.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> [time_interval]\n";
        return -1;
    }

    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load images\n";
        return -1;
    }

    float timeInterval = (argc >= 4) ? std::stof(argv[3]) : 0.5f;

    MovingObject::MotionDetector detector(1000, 0.7f, 15);
    cv::Mat motionMask;
    cv::Mat result = detector.DetectMotion(img1, img2, motionMask, timeInterval);

    cv::imshow("Motion Detection", result);
    cv::imshow("Motion Mask", motionMask);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
