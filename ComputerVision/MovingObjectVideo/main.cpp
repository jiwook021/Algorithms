/**
 * @file main.cpp
 * @brief Driver for MovingObjectVideo module
 */

#include "MovingObjectVideo.hpp"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file> [frame_skip]\n";
        return -1;
    }

    std::string videoPath = argv[1];
    int frameSkip = (argc >= 3) ? std::stoi(argv[2]) : 1;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video: " << videoPath << "\n";
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    float timeInterval = frameSkip / static_cast<float>(fps);
    std::cout << "FPS: " << fps << "  dt: " << timeInterval << "s\n";

    MovingObjectVideo::MotionDetector detector(1000, 0.7f, 15);

    cv::Mat prevFrame, currFrame;
    cap.read(prevFrame);
    for (int i = 0; i < frameSkip - 1 && cap.read(currFrame); ++i) {}

    while (cap.read(currFrame)) {
        for (int i = 0; i < frameSkip - 1 && cap.read(currFrame); ++i) {}

        cv::Mat motionMask;
        cv::Mat result = detector.DetectMotion(prevFrame, currFrame, motionMask, timeInterval);

        cv::imshow("Motion Detection", result);
        prevFrame = currFrame.clone();

        int key = cv::waitKey(30);
        if (key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
