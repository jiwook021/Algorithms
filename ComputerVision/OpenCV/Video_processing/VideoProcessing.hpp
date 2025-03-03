/**
 * @file VideoProcessing.hpp
 * @brief Video frame processing pipeline (read, transform, write)
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <functional>
namespace VideoProcessing {
using FrameProcessor = std::function<cv::Mat(const cv::Mat&)>;
cv::Mat ConvertToGrayBgr(const cv::Mat& frame);
bool ProcessVideo(const std::string& input, const std::string& output,
                  FrameProcessor processor);
}
