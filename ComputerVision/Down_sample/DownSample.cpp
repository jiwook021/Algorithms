/**
 * @file DownSample.cpp
 * @brief Implementation of image downsampling
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "DownSample.hpp"
#include <stdexcept>
#include <algorithm>

namespace DownSample {

Image::Image(const std::string& filename) {
    int w, h, c;
    unsigned char* data = stbi_load(filename.c_str(), &w, &h, &c, CHANNELS);
    if (!data) throw std::runtime_error("Failed to load image: " + filename);
    width_  = w;
    height_ = h;
    pixels_.assign(data, data + width_ * height_ * CHANNELS);
    stbi_image_free(data);
}

Image::Image(int width, int height)
    : width_(width), height_(height), pixels_(width * height * CHANNELS, 0) {}

bool Image::Save(const std::string& filename) const {
    return stbi_write_png(filename.c_str(), width_, height_, CHANNELS,
                          pixels_.data(), width_ * CHANNELS);
}

const uint8_t* Image::PixelAt(int x, int y) const {
    return &pixels_[(y * width_ + x) * CHANNELS];
}

uint8_t* Image::PixelAt(int x, int y) {
    return &pixels_[(y * width_ + x) * CHANNELS];
}

void Image::ComputeAvgPixel(int dstX, int dstY, const Image& src) {
    int srcX = dstX * 2;
    int srcY = dstY * 2;
    int sum[CHANNELS] = {};
    int count = 0;

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int ix = srcX + dx;
            int iy = srcY + dy;
            if (ix < src.width_ && iy < src.height_) {
                const uint8_t* p = src.PixelAt(ix, iy);
                for (int c = 0; c < CHANNELS; ++c) sum[c] += p[c];
                ++count;
            }
        }
    }

    uint8_t* dst = PixelAt(dstX, dstY);
    for (int c = 0; c < CHANNELS; ++c)
        dst[c] = static_cast<uint8_t>(sum[c] / count);
}

std::unique_ptr<Image> Image::HalfSize() const {
    int newWidth  = width_  / 2;
    int newHeight = height_ / 2;
    auto result = std::make_unique<Image>(newWidth, newHeight);

    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    std::vector<std::thread> threads;
    int rowsPerThread = newHeight / static_cast<int>(numThreads);
    int extra         = newHeight % static_cast<int>(numThreads);

    auto worker = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y)
            for (int x = 0; x < newWidth; ++x)
                result->ComputeAvgPixel(x, y, *this);
    };

    int currentY = 0;
    for (unsigned i = 0; i < numThreads; ++i) {
        int startY = currentY;
        int endY   = startY + rowsPerThread + (static_cast<int>(i) < extra ? 1 : 0);
        threads.emplace_back(worker, startY, endY);
        currentY = endY;
    }
    for (auto& t : threads) t.join();
    return result;
}

}  // namespace DownSample
