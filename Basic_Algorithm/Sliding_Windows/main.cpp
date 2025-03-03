/**
 * @file main.cpp
 * @brief Driver for SlidingWindows
 */

#include "SlidingWindows.hpp"
#include <iostream>

int main() {
    std::vector<int> arr = {1, 4, 2, 10, 2, 3, 1, 0, 20};
    int k = 4;

    int result = SlidingWindows::MaxSumSubarray(arr, k);
    std::cout << "Maximum sum of subarray with size " << k << ": "
              << result << std::endl;

    return 0;
}
