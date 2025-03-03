/**
 * @file GaussianBlur.hpp
 * @brief Separable Gaussian blur filter (no OpenCV dependency)
 * @details Uses stb_image for I/O. Generates a 1D Gaussian kernel and applies
 *          it in two passes (horizontal, vertical) for O(n*k) per pixel.
 */

#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

#include "stb_image.h"
#include "stb_image_write.h"

namespace GaussianBlur {

class Image {
public:
    Image();
    Image(int width, int height, int channels);
    Image(const Image& other);

    bool Load(const std::string& filename);
    bool Save(const std::string& filename) const;

    int GetWidth()    const { return width_; }
    int GetHeight()   const { return height_; }
    int GetChannels() const { return channels_; }

    bool GetPixel(int x, int y, unsigned char* pixel) const;
    bool SetPixel(int x, int y, const unsigned char* pixel);

private:
    int width_;
    int height_;
    int channels_;
    std::vector<unsigned char> data_;
};

/// Generate a normalised 1D Gaussian kernel for the given sigma.
std::vector<double> GenerateGaussianKernel(double sigma);

/// Apply separable Gaussian blur to an Image.
Image ApplyGaussianBlur(const Image& input, double sigma);

}  // namespace GaussianBlur
