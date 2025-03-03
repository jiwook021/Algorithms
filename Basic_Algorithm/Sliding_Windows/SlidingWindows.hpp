/**
 * @file SlidingWindows.hpp
 * @brief Sliding window maximum subarray sum
 * @details Finds the maximum sum of a contiguous subarray of size k
 *          using the sliding window technique in O(n) time.
 */

#pragma once

#include <vector>
#include <algorithm>

namespace SlidingWindows {

/**
 * @brief Find maximum sum of a contiguous subarray of size k.
 * @param arr Input array
 * @param k   Window size
 * @return Maximum sum, or -1 if array size < k
 */
int MaxSumSubarray(const std::vector<int>& arr, int k);

} // namespace SlidingWindows
