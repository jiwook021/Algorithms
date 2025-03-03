/**
 * @file main.cpp
 * @brief Driver for FirIir module -- generates signal, filters, and plots with OpenCV
 */

#include "FirIir.hpp"
#include <opencv2/opencv.hpp>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>

int main() {
    try {
        constexpr size_t NUM_SAMPLES = 200;
        std::vector<double> signal(NUM_SAMPLES);
        std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::normal_distribution<double> dist(0.0, 1.0);
        std::generate(signal.begin(), signal.end(), [&]() { return dist(rng); });

        FirIir::FirFilter fir({0.1, 0.15, 0.5, 0.15, 0.1});
        FirIir::IirFilter iir(0.1);

        std::vector<double> firOut, iirOut;
        firOut.reserve(NUM_SAMPLES);
        iirOut.reserve(NUM_SAMPLES);
        for (double s : signal) {
            firOut.push_back(fir.Process(s));
            iirOut.push_back(iir.Process(s));
        }

        auto allMin = [](auto& a, auto& b, auto& c) {
            return std::min({*std::min_element(a.begin(), a.end()),
                             *std::min_element(b.begin(), b.end()),
                             *std::min_element(c.begin(), c.end())});
        };
        auto allMax = [](auto& a, auto& b, auto& c) {
            return std::max({*std::max_element(a.begin(), a.end()),
                             *std::max_element(b.begin(), b.end()),
                             *std::max_element(c.begin(), c.end())});
        };
        double minVal = allMin(signal, firOut, iirOut);
        double maxVal = allMax(signal, firOut, iirOut);
        if (maxVal == minVal) { maxVal += 1; minVal -= 1; }

        constexpr int W = 800, H = 600;
        cv::Mat plot(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

        auto toPoint = [&](size_t i, double v) {
            int x = static_cast<int>(double(i) / (NUM_SAMPLES - 1) * (W - 1));
            int y = static_cast<int>((1.0 - (v - minVal) / (maxVal - minVal)) * (H - 1));
            return cv::Point(x, y);
        };

        std::vector<cv::Point> ptsSig, ptsFir, ptsIir;
        for (size_t i = 0; i < NUM_SAMPLES; ++i) {
            ptsSig.push_back(toPoint(i, signal[i]));
            ptsFir.push_back(toPoint(i, firOut[i]));
            ptsIir.push_back(toPoint(i, iirOut[i]));
        }

        cv::polylines(plot, ptsSig, false, cv::Scalar(255, 0, 0), 2);
        cv::polylines(plot, ptsFir, false, cv::Scalar(0, 255, 0), 2);
        cv::polylines(plot, ptsIir, false, cv::Scalar(0, 0, 255), 2);

        cv::putText(plot, "Original",     {30, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
        cv::putText(plot, "FIR Filtered", {30, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        cv::putText(plot, "IIR Filtered", {30, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);

        cv::imshow("Signal vs FIR vs IIR", plot);
        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
