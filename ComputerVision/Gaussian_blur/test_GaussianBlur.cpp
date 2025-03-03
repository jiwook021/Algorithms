/**
 * @file test_GaussianBlur.cpp
 * @brief Unit tests for GaussianBlur module
 */

#include "GaussianBlur.hpp"
#include <cassert>
#include <cmath>
#include <numeric>

namespace {

void TestKernelSumsToOne() {
    auto k = GaussianBlur::GenerateGaussianKernel(1.0);
    double sum = 0;
    for (auto v : k) sum += v;
    assert(std::abs(sum - 1.0) < 1e-6);
    std::cout << "  TestKernelSumsToOne PASSED\n";
}

void TestKernelIsOddLength() {
    for (double sigma : {0.5, 1.0, 2.0, 3.5}) {
        auto k = GaussianBlur::GenerateGaussianKernel(sigma);
        assert(k.size() % 2 == 1);
    }
    std::cout << "  TestKernelIsOddLength PASSED\n";
}

void TestKernelSymmetry() {
    auto k = GaussianBlur::GenerateGaussianKernel(2.0);
    int n = static_cast<int>(k.size());
    for (int i = 0; i < n / 2; ++i)
        assert(std::abs(k[i] - k[n - 1 - i]) < 1e-10);
    std::cout << "  TestKernelSymmetry PASSED\n";
}

void TestImageCreateAndPixels() {
    GaussianBlur::Image img(4, 4, 3);
    unsigned char px[3] = {100, 150, 200};
    assert(img.SetPixel(2, 3, px));
    unsigned char out[3] = {};
    assert(img.GetPixel(2, 3, out));
    assert(out[0] == 100 && out[1] == 150 && out[2] == 200);
    std::cout << "  TestImageCreateAndPixels PASSED\n";
}

void TestBlurPreservesUniformImage() {
    GaussianBlur::Image img(8, 8, 1);
    unsigned char val = 128;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            img.SetPixel(x, y, &val);

    auto blurred = GaussianBlur::ApplyGaussianBlur(img, 1.0);

    unsigned char px;
    blurred.GetPixel(4, 4, &px);
    assert(px == 128);
    std::cout << "  TestBlurPreservesUniformImage PASSED\n";
}

void TestBlurReducesContrast() {
    // Create 10x10 image with a bright spot in center
    GaussianBlur::Image img(10, 10, 1);
    unsigned char bg = 0;
    for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 10; ++x)
            img.SetPixel(x, y, &bg);
    unsigned char bright = 255;
    img.SetPixel(5, 5, &bright);

    auto blurred = GaussianBlur::ApplyGaussianBlur(img, 1.5);

    unsigned char centerVal;
    blurred.GetPixel(5, 5, &centerVal);
    // Blurring should reduce the center brightness
    assert(centerVal < 255);
    std::cout << "  TestBlurReducesContrast PASSED\n";
}

void TestBoundsCheck() {
    GaussianBlur::Image img(2, 2, 1);
    unsigned char px;
    assert(!img.GetPixel(-1, 0, &px));
    assert(!img.GetPixel(0, 2, &px));
    assert(!img.SetPixel(3, 0, &px));
    std::cout << "  TestBoundsCheck PASSED\n";
}

}  // namespace

int main() {
    std::cout << "Running GaussianBlur tests...\n";
    TestKernelSumsToOne();
    TestKernelIsOddLength();
    TestKernelSymmetry();
    TestImageCreateAndPixels();
    TestBlurPreservesUniformImage();
    TestBlurReducesContrast();
    TestBoundsCheck();
    std::cout << "All GaussianBlur tests PASSED.\n";
    return 0;
}
