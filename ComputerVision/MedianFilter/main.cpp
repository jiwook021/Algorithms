/**
 * @file main.cpp
 * @brief Driver for MedianFilter module
 */

#include "MedianFilter.hpp"
#include <iostream>
#include <vector>

int main() {
    constexpr int W = 10;
    constexpr int H = 10;

    std::vector<int> src(W * H, 1);
    // Add diagonal noise
    for (int i = 1; i < H - 1; ++i)
        src[i * W + i] = 1000;

    std::vector<int> dst(W * H, 0);

    MedianFilter::ApplyThreaded(src.data(), dst.data(), W, H, 3);

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j)
            std::cout << dst[i * W + j] << " ";
        std::cout << "\n";
    }
    return 0;
}
