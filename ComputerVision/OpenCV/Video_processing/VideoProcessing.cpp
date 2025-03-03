#include "VideoProcessing.hpp"
#include <iostream>
namespace VideoProcessing {
cv::Mat ConvertToGrayBgr(const cv::Mat& frame) {
    cv::Mat gray, bgr;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}
bool ProcessVideo(const std::string& input, const std::string& output,
                  FrameProcessor processor) {
    cv::VideoCapture cap(input);
    if (!cap.isOpened()) return false;
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(output, cv::VideoWriter::fourcc('M','J','P','G'),
                           fps, cv::Size(w, h));
    if (!writer.isOpened()) return false;
    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat processed = processor(frame);
        writer.write(processed);
    }
    cap.release();
    writer.release();
    return true;
}
}
