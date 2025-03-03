/**
 * @file FIR_IIR.hpp
 * @brief FIR and IIR filter implementations (Docker variant)
 * @details Identical FIR/IIR pipeline configured for headless Docker execution. Uses file I/O instead of cv::imshow for output.
 */

#pragma once

// random_signal_filters_opencv.cpp
// C++23 | Requires OpenCV (e.g., `pkg-config --cflags --libs opencv4`)
// Generates a random signal, applies an FIR filter and a 1st-order IIR filter,
// and plots original vs. filtered signals using OpenCV.

#include <iostream>           // std::cout, std::cerr
#include <vector>             // std::vector
#include <random>             // std::mt19937_64, std::normal_distribution
#include <chrono>             // std::chrono::high_resolution_clock
#include <stdexcept>          // std::invalid_argument, std::exception
#include <algorithm>          // std::generate, std::min_element, std::max_element

#include <opencv2/opencv.hpp> // Core OpenCV functionality

// ----------------------------------------
// FIRFilter: Finite Impulse Response
// - Time complexity per sample: O(M) where M = number of taps
// - Memory complexity: O(M)
// ----------------------------------------

class FIRFilter {
public:
    explicit FIRFilter(const std::vector<double>& Taps)
        : Taps_(Taps)
    {
        if (Taps_.empty()) {
            // throw exception if no coefficients provided
            throw std::invalid_argument("FIRFilter: taps vector must not be empty");
        }
        // initialize circular buffer with zeros
        Buffer_.assign(Taps_.size(), 0.0);
        WriteIndex_ = 0;
    }

    // Process one sample and return filtered output
    double Process(double Input) {
        Buffer_[WriteIndex_] = Input;  // write newest sample into buffer
        double Output = 0.0;
        size_t Idx = WriteIndex_;
        // convolution: sum taps[k] * buffer_[idx]
        for (size_t k = 0; k < Taps_.size(); ++k) {
            Output += Taps_[k] * Buffer_[Idx];
            // move idx backwards in circular fashion
            Idx = (Idx == 0) ? Buffer_.size() - 1 : Idx - 1;
        }
        // advance writeIndex_ circularly
        WriteIndex_ = (WriteIndex_ + 1) % Buffer_.size();
        return Output;
    }

private:
    std::vector<double> Taps_;
    std::vector<double> Buffer_;
    size_t WriteIndex_;
};

// ----------------------------------------
// IIRFilter: First-order exponential smoothing
// - y[n] = α·x[n] + (1-α)·y[n-1]
// - Time complexity per sample: O(1)
// - Memory complexity: O(1)
// ----------------------------------------
class IIRFilter {
public:
    explicit IIRFilter(double Alpha)
        : Alpha_(Alpha), PrevOutput_(0.0), Initialized_(false)
    {
        if (!(Alpha > 0.0 && Alpha < 1.0)) {
            // ensure alpha in (0,1)
            throw std::invalid_argument("IIRFilter: alpha must be in (0,1)");
        }
    }

    // Process one sample and return filtered output
    double Process(double Input) {
        if (!Initialized_) {
            // on first call, initialize prevOutput_ to the first input
            PrevOutput_ = Input;
            Initialized_ = true;
        }
        // apply exponential smoothing
        double Output = Alpha_ * Input + (1.0 - Alpha_) * PrevOutput_;
        PrevOutput_ = Output;
        return Output;
    }

private:
    double Alpha_;
    double PrevOutput_;
    bool Initialized_;
};

