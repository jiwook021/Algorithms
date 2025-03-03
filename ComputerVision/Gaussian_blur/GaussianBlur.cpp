/**
 * @file GaussianBlur.cpp
 * @brief Implementation of separable Gaussian blur
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "GaussianBlur.hpp"

namespace GaussianBlur {

// ---- Image implementation ----

Image::Image() : width_(0), height_(0), channels_(0) {}

Image::Image(int w, int h, int c)
    : width_(w), height_(h), channels_(c), data_(w * h * c, 0) {}

Image::Image(const Image& other)
    : width_(other.width_), height_(other.height_),
      channels_(other.channels_), data_(other.data_) {}

bool Image::Load(const std::string& filename) {
    data_.clear();
    int w, h, c;
    unsigned char* imgData = stbi_load(filename.c_str(), &w, &h, &c, 0);
    if (!imgData) {
        std::cerr << "Error loading image: " << filename << std::endl;
        return false;
    }
    width_    = w;
    height_   = h;
    channels_ = c;
    data_.assign(imgData, imgData + (width_ * height_ * channels_));
    stbi_image_free(imgData);
    return true;
}

bool Image::Save(const std::string& filename) const {
    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    if (ext == "jpg" || ext == "jpeg")
        return stbi_write_jpg(filename.c_str(), width_, height_, channels_,
                              data_.data(), 90);
    if (ext == "png")
        return stbi_write_png(filename.c_str(), width_, height_, channels_,
                              data_.data(), width_ * channels_);
    std::cerr << "Unsupported format: " << ext << std::endl;
    return false;
}

bool Image::GetPixel(int x, int y, unsigned char* pixel) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) return false;
    int idx = (y * width_ + x) * channels_;
    for (int c = 0; c < channels_; ++c) pixel[c] = data_[idx + c];
    return true;
}

bool Image::SetPixel(int x, int y, const unsigned char* pixel) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) return false;
    int idx = (y * width_ + x) * channels_;
    for (int c = 0; c < channels_; ++c) data_[idx + c] = pixel[c];
    return true;
}

// ---- Gaussian kernel ----

std::vector<double> GenerateGaussianKernel(double sigma) {
    int kernelSize = static_cast<int>(std::ceil(sigma * 6));
    if (kernelSize % 2 == 0) ++kernelSize;

    std::vector<double> kernel(kernelSize);
    double sum    = 0.0;
    int    center = kernelSize / 2;

    for (int i = 0; i < kernelSize; ++i) {
        int x = i - center;
        kernel[i] = std::exp(-(x * x) / (2.0 * sigma * sigma))
                    / (std::sqrt(2.0 * M_PI) * sigma);
        sum += kernel[i];
    }
    for (auto& v : kernel) v /= sum;
    return kernel;
}

// ---- Blur ----

Image ApplyGaussianBlur(const Image& input, double sigma) {
    int width    = input.GetWidth();
    int height   = input.GetHeight();
    int channels = input.GetChannels();

    auto kernel       = GenerateGaussianKernel(sigma);
    int  kernelSize   = static_cast<int>(kernel.size());
    int  kernelRadius = kernelSize / 2;

    Image temp(width, height, channels);
    Image output(width, height, channels);

    std::vector<unsigned char> inPx(channels);
    std::vector<unsigned char> outPx(channels);

    // Horizontal pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<double> sums(channels, 0.0);
            double weightSum = 0.0;
            for (int i = 0; i < kernelSize; ++i) {
                int sampleX = x + (i - kernelRadius);
                if (input.GetPixel(sampleX, y, inPx.data())) {
                    for (int c = 0; c < channels; ++c)
                        sums[c] += inPx[c] * kernel[i];
                    weightSum += kernel[i];
                }
            }
            if (weightSum > 0) {
                for (int c = 0; c < channels; ++c)
                    outPx[c] = static_cast<unsigned char>(sums[c] / weightSum);
                temp.SetPixel(x, y, outPx.data());
            }
        }
    }

    // Vertical pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<double> sums(channels, 0.0);
            double weightSum = 0.0;
            for (int i = 0; i < kernelSize; ++i) {
                int sampleY = y + (i - kernelRadius);
                if (temp.GetPixel(x, sampleY, inPx.data())) {
                    for (int c = 0; c < channels; ++c)
                        sums[c] += inPx[c] * kernel[i];
                    weightSum += kernel[i];
                }
            }
            if (weightSum > 0) {
                for (int c = 0; c < channels; ++c)
                    outPx[c] = static_cast<unsigned char>(sums[c] / weightSum);
                output.SetPixel(x, y, outPx.data());
            }
        }
    }
    return output;
}

}  // namespace GaussianBlur
