/**
 * @file MedianFilter.cpp
 * @brief Implementation of the median filter
 */

#include "MedianFilter.hpp"
#include <thread>
#include <cstring>

namespace MedianFilter {

static void FilterRows(const int* src, int* dst,
                       int width, int height,
                       int kernelSize,
                       int startRow, int endRow) {
    int k = kernelSize / 2;

    // Copy border rows unchanged
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < width; ++j) {
            if (i < k || i >= height - k || j < k || j >= width - k) {
                dst[i * width + j] = src[i * width + j];
            }
        }
    }

    // Median filter interior
    for (int i = std::max(startRow, k); i < std::min(endRow, height - k); ++i) {
        for (int j = k; j < width - k; ++j) {
            std::vector<int> window;
            window.reserve(kernelSize * kernelSize);
            for (int dy = -k; dy <= k; ++dy)
                for (int dx = -k; dx <= k; ++dx)
                    window.push_back(src[(i + dy) * width + (j + dx)]);

            std::nth_element(window.begin(),
                             window.begin() + static_cast<int>(window.size()) / 2,
                             window.end());
            dst[i * width + j] = window[window.size() / 2];
        }
    }
}

void Apply(const int* src, int* dst,
           int width, int height, int kernelSize) {
    FilterRows(src, dst, width, height, kernelSize, 0, height);
}

void ApplyThreaded(const int* src, int* dst,
                   int width, int height, int kernelSize,
                   int threadCount) {
    if (threadCount <= 0)
        threadCount = static_cast<int>(std::thread::hardware_concurrency());
    if (threadCount <= 0) threadCount = 1;

    std::vector<std::thread> workers;
    int rowsPerThread = height / threadCount;

    for (int t = 0; t < threadCount; ++t) {
        int startRow = t * rowsPerThread;
        int endRow   = (t == threadCount - 1) ? height : (t + 1) * rowsPerThread;
        workers.emplace_back(FilterRows, src, dst, width, height,
                             kernelSize, startRow, endRow);
    }
    for (auto& w : workers) w.join();
}

}  // namespace MedianFilter
