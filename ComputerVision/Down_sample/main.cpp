#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include "stb_image.h"
#include "stb_image_write.h"

class Image {
public:
    Image(const std::string& filename);
    Image(int w, int h);

    bool Save(const std::string& filename) const;
    std::unique_ptr<Image> DownSample() const;

private:
    int width_;
    int height_;
    int channels_;
    std::vector<uint8_t> pixels_; // RGBA pixels

    void ComputeAvgPixel(int dstX, int dstY, const Image& src);
};

Image::Image(const std::string& filename) {
    unsigned char* data = stbi_load(filename.c_str(), &width_, &height_, &channels_, 4);
    if (!data) throw std::runtime_error("Failed to load image: " + filename);
    channels_ = 4;
    pixels_.assign(data, data + width_ * height_ * channels_);
    stbi_image_free(data);
}

Image::Image(int w, int h) : width_(w), height_(h), channels_(4), pixels_(w * h * 4) {}

bool Image::Save(const std::string& filename) const {
    return stbi_write_png(filename.c_str(), width_, height_, channels_, pixels_.data(), width_ * channels_);
}

void Image::ComputeAvgPixel(int dstX, int dstY, const Image& src) {
    int srcX = dstX * 2;
    int srcY = dstY * 2;
    int sum[4] = {0, 0, 0, 0};
    int count = 0;

    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int ix = srcX + dx;
            int iy = srcY + dy;
            if (ix < src.width_ && iy < src.height_) {
                const uint8_t* p = &src.pixels_[(iy * src.width_ + ix) * 4];
                for (int c = 0; c < 4; c++)
                    sum[c] += p[c];
                count++;
            }
        }
    }

    uint8_t* dstPixel = &pixels_[(dstY * width_ + dstX) * 4];
    for (int c = 0; c < 4; c++)
        dstPixel[c] = static_cast<uint8_t>(sum[c] / count);
}

std::unique_ptr<Image> Image::DownSample() const {
    int newWidth = width_ / 2;
    int newHeight = height_ / 2;
    auto result = std::make_unique<Image>(newWidth, newHeight);

    unsigned numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;

    std::vector<std::thread> threads;
    int rowsPerThread = newHeight / numThreads;
    int extra = newHeight % numThreads;

    auto worker = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y)
            for (int x = 0; x < newWidth; ++x)
                result->ComputeAvgPixel(x, y, *this);
    };

    int currentY = 0;
    for (unsigned i = 0; i < numThreads; ++i) {
        int startY = currentY;
        int endY = startY + rowsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back(worker, startY, endY);
        currentY = endY;
    }

    for (auto& t : threads) t.join();
    return result;
}

int main() {
    try {
        Image img("input.png");
        auto downsampled = img.DownSample();
        if (downsampled->Save("output.png"))
            std::cout << "Saved successfully!\n";
        else
            std::cerr << "Failed to save image!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return -1;
    }
    return 0;
}
