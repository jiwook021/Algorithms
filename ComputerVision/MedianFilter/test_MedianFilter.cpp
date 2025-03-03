/**
 * @file test_MedianFilter.cpp
 * @brief Unit tests for MedianFilter module
 */

#include "MedianFilter.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>

namespace {

void TestUniformImage() {
    constexpr int W = 5, H = 5;
    std::vector<int> src(W * H, 42);
    std::vector<int> dst(W * H, 0);
    MedianFilter::Apply(src.data(), dst.data(), W, H, 3);
    for (int i = 0; i < W * H; ++i)
        assert(dst[i] == 42);
    std::cout << "  TestUniformImage PASSED\n";
}

void TestSaltNoiseSinglePixel() {
    constexpr int W = 5, H = 5;
    std::vector<int> src(W * H, 10);
    src[2 * W + 2] = 1000;  // Salt noise at center
    std::vector<int> dst(W * H, 0);
    MedianFilter::Apply(src.data(), dst.data(), W, H, 3);
    // Center should become 10 (median of eight 10s and one 1000)
    assert(dst[2 * W + 2] == 10);
    std::cout << "  TestSaltNoiseSinglePixel PASSED\n";
}

void TestBorderPreserved() {
    constexpr int W = 5, H = 5;
    std::vector<int> src(W * H, 0);
    src[0]          = 99;  // top-left corner
    src[W - 1]      = 88;  // top-right corner
    src[(H-1) * W]  = 77;  // bottom-left corner
    std::vector<int> dst(W * H, 0);
    MedianFilter::Apply(src.data(), dst.data(), W, H, 3);
    assert(dst[0]          == 99);
    assert(dst[W - 1]      == 88);
    assert(dst[(H-1) * W]  == 77);
    std::cout << "  TestBorderPreserved PASSED\n";
}

void TestKnownMedian() {
    // 3x3 image; kernel=3 covers entire interior
    constexpr int W = 3, H = 3;
    int src[] = {1, 2, 3,
                 4, 5, 6,
                 7, 8, 9};
    int dst[W * H] = {};
    MedianFilter::Apply(src, dst, W, H, 3);
    // Center pixel: median of {1..9} = 5
    assert(dst[1 * W + 1] == 5);
    std::cout << "  TestKnownMedian PASSED\n";
}

void TestThreadedMatchesSingleThread() {
    constexpr int W = 8, H = 8;
    std::vector<int> src(W * H);
    for (int i = 0; i < W * H; ++i) src[i] = i % 17;

    std::vector<int> dstSingle(W * H, 0);
    std::vector<int> dstThreaded(W * H, 0);

    MedianFilter::Apply(src.data(), dstSingle.data(), W, H, 3);
    MedianFilter::ApplyThreaded(src.data(), dstThreaded.data(), W, H, 3, 4);

    assert(dstSingle == dstThreaded);
    std::cout << "  TestThreadedMatchesSingleThread PASSED\n";
}

void TestDimensions() {
    // Larger image
    constexpr int W = 20, H = 15;
    std::vector<int> src(W * H, 50);
    src[7 * W + 10] = 999;
    std::vector<int> dst(W * H, 0);
    MedianFilter::Apply(src.data(), dst.data(), W, H, 5);
    // Single spike should be removed by 5x5 kernel
    assert(dst[7 * W + 10] == 50);
    std::cout << "  TestDimensions PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running MedianFilter tests...\n";
    TestUniformImage();
    TestSaltNoiseSinglePixel();
    TestBorderPreserved();
    TestKnownMedian();
    TestThreadedMatchesSingleThread();
    TestDimensions();
    std::cout << "All MedianFilter tests PASSED.\n";
    return 0;
}
