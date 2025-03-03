/**
 * @file MedianFilter.hpp
 * @brief Median filter for salt-and-pepper noise removal
 * @details Sliding-window median filter operating on a 2D integer array.
 *          Uses std::nth_element for O(k^2) median selection per pixel.
 */

#pragma once

#include <vector>
#include <algorithm>

namespace MedianFilter {

/**
 * @brief Apply a median filter to a 2D grayscale buffer.
 * @param src          Input buffer (row-major, height rows x width cols).
 * @param dst          Output buffer (same dimensions, caller must allocate).
 * @param width        Number of columns.
 * @param height       Number of rows.
 * @param kernelSize   Side length of the square kernel (must be odd, >= 3).
 *
 * Border pixels are copied unchanged from src.
 */
void Apply(const int* src, int* dst,
           int width, int height, int kernelSize = 3);

/**
 * @brief Multithreaded variant -- splits rows across threads.
 */
void ApplyThreaded(const int* src, int* dst,
                   int width, int height, int kernelSize = 3,
                   int threadCount = 0);

}  // namespace MedianFilter
