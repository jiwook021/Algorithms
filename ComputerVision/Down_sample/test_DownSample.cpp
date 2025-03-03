/**
 * @file test_DownSample.cpp
 * @brief Unit tests for DownSample module
 */

#include "DownSample.hpp"
#include <cassert>
#include <iostream>
#include <cstring>

namespace {

void TestConstructFromDimensions() {
    DownSample::Image img(10, 8);
    assert(img.GetWidth()  == 10);
    assert(img.GetHeight() == 8);
    // All pixels should be zero-initialized
    const uint8_t* px = img.PixelAt(0, 0);
    assert(px[0] == 0 && px[1] == 0 && px[2] == 0 && px[3] == 0);
    std::cout << "  TestConstructFromDimensions PASSED\n";
}

void TestHalfSizeDimensions() {
    DownSample::Image img(20, 16);
    auto half = img.HalfSize();
    assert(half->GetWidth()  == 10);
    assert(half->GetHeight() == 8);
    std::cout << "  TestHalfSizeDimensions PASSED\n";
}

void TestHalfSizeAveraging() {
    // 4x4 image: set all pixels to known RGBA values
    DownSample::Image img(4, 4);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            uint8_t* px = img.PixelAt(x, y);
            px[0] = 100;  // R
            px[1] = 200;  // G
            px[2] = 50;   // B
            px[3] = 255;  // A
        }
    }

    auto half = img.HalfSize();
    assert(half->GetWidth()  == 2);
    assert(half->GetHeight() == 2);

    // Average of identical pixels should be the same value
    const uint8_t* px = half->PixelAt(0, 0);
    assert(px[0] == 100);
    assert(px[1] == 200);
    assert(px[2] == 50);
    assert(px[3] == 255);
    std::cout << "  TestHalfSizeAveraging PASSED\n";
}

void TestHalfSizeMixedPixels() {
    // 2x2 image with different values
    DownSample::Image img(2, 2);
    // (0,0) = {0,0,0,0}, (1,0) = {100,100,100,100}
    // (0,1) = {200,200,200,200}, (1,1) = {100,100,100,100}
    uint8_t* p00 = img.PixelAt(0, 0);
    p00[0] = 0; p00[1] = 0; p00[2] = 0; p00[3] = 0;

    uint8_t* p10 = img.PixelAt(1, 0);
    p10[0] = 100; p10[1] = 100; p10[2] = 100; p10[3] = 100;

    uint8_t* p01 = img.PixelAt(0, 1);
    p01[0] = 200; p01[1] = 200; p01[2] = 200; p01[3] = 200;

    uint8_t* p11 = img.PixelAt(1, 1);
    p11[0] = 100; p11[1] = 100; p11[2] = 100; p11[3] = 100;

    auto half = img.HalfSize();
    const uint8_t* avg = half->PixelAt(0, 0);
    // Average = (0+100+200+100)/4 = 100
    assert(avg[0] == 100);
    assert(avg[1] == 100);
    std::cout << "  TestHalfSizeMixedPixels PASSED\n";
}

void TestOddDimensions() {
    // 5x5 image - half should be 2x2 (integer division)
    DownSample::Image img(5, 5);
    auto half = img.HalfSize();
    assert(half->GetWidth()  == 2);
    assert(half->GetHeight() == 2);
    std::cout << "  TestOddDimensions PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running DownSample tests...\n";
    TestConstructFromDimensions();
    TestHalfSizeDimensions();
    TestHalfSizeAveraging();
    TestHalfSizeMixedPixels();
    TestOddDimensions();
    std::cout << "All DownSample tests PASSED.\n";
    return 0;
}
