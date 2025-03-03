/**
 * @file SlidingWindows.cpp
 * @brief Implementation of sliding window algorithms
 */

#include "SlidingWindows.hpp"

namespace SlidingWindows {

int MaxSumSubarray(const std::vector<int>& arr, int k) {
    if (static_cast<int>(arr.size()) < k || k <= 0) {
        return -1;
    }

    int windowSum = 0;
    for (int i = 0; i < k; ++i) {
        windowSum += arr[i];
    }

    int maxSum = windowSum;

    for (int i = k; i < static_cast<int>(arr.size()); ++i) {
        windowSum += arr[i] - arr[i - k];
        maxSum = std::max(maxSum, windowSum);
    }

    return maxSum;
}

} // namespace SlidingWindows
