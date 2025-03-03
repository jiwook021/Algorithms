/**
 * @file DownSample.hpp
 * @brief Image downsampling using 2x2 block averaging with multithreading
 */

#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <string>
#include <cstdint>

// Forward-declare stb functions so the header stays include-free.
// STB_IMAGE_IMPLEMENTATION / STB_IMAGE_WRITE_IMPLEMENTATION are defined
// only in DownSample.cpp to avoid multiple-definition errors.
#include "stb_image.h"
#include "stb_image_write.h"

namespace DownSample {

class Image {
public:
    /// Load image from file (always stored as RGBA).
    explicit Image(const std::string& filename);

    /// Create blank RGBA image.
    Image(int width, int height);

    /// Save image to PNG file.
    bool Save(const std::string& filename) const;

    /// Produce a new image at half resolution using 2x2 block averaging.
    std::unique_ptr<Image> HalfSize() const;

    int GetWidth()    const { return width_; }
    int GetHeight()   const { return height_; }
    int GetChannels() const { return CHANNELS; }

    /// Direct pixel access (RGBA, 4 bytes per pixel).
    const uint8_t* PixelAt(int x, int y) const;
    uint8_t*       PixelAt(int x, int y);

private:
    static constexpr int CHANNELS = 4;
    int width_;
    int height_;
    std::vector<uint8_t> pixels_;

    void ComputeAvgPixel(int dstX, int dstY, const Image& src);
};

}  // namespace DownSample
