#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

// Include the stb_image libraries for image loading and saving
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Class representing an RGB image
class Image {
private:
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data; // Stores RGB values sequentially

public:
    // Default constructor
    Image() : width(0), height(0), channels(0) {}
    
    // Constructor for creating an image with specific dimensions
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(width * height * channels, 0);
    }
    
    // Copy constructor
    Image(const Image& other) : width(other.width), height(other.height), 
                               channels(other.channels), data(other.data) {}
    
    // Load image from file
    bool load(const std::string& filename) {
        // Free any existing image data
        if (!data.empty()) {
            data.clear();
        }
        
        // Load image using stb_image
        int w, h, c;
        unsigned char* imgData = stbi_load(filename.c_str(), &w, &h, &c, 0);
        
        if (!imgData) {
            std::cerr << "Error loading image: " << filename << std::endl;
            return false;
        }
        
        // Set image properties
        width = w;
        height = h;
        channels = c;
        
        // Copy image data to our vector
        data.assign(imgData, imgData + (width * height * channels));
        
        // Free the loaded image data
        stbi_image_free(imgData);
        
        std::cout << "Loaded image: " << filename << " (" << width << "x" 
                  << height << ", " << channels << " channels)" << std::endl;
        return true;
    }
    
    // Save image to file
    bool save(const std::string& filename) const {
        // Determine file format from extension
        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        
        bool success = false;
        if (ext == "jpg" || ext == "jpeg") {
            success = stbi_write_jpg(filename.c_str(), width, height, channels, 
                                   data.data(), 90); // Quality parameter (0-100)
        } else if (ext == "png") {
            success = stbi_write_png(filename.c_str(), width, height, channels, 
                                   data.data(), width * channels);
        } else {
            std::cerr << "Unsupported image format: " << ext << std::endl;
            return false;
        }
        
        if (!success) {
            std::cerr << "Error saving image: " << filename << std::endl;
            return false;
        }
        
        std::cout << "Saved image: " << filename << std::endl;
        return true;
    }
    
    // Getters for image dimensions
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }
    
    // Access pixel at (x,y) with bounds checking
    bool getPixel(int x, int y, unsigned char* pixel) const {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return false;
            
        int index = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            pixel[c] = data[index + c];
        }
        return true;
    }
    
    // Set pixel at (x,y) with bounds checking
    bool setPixel(int x, int y, const unsigned char* pixel) {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return false;
            
        int index = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            data[index + c] = pixel[c];
        }
        return true;
    }
};

// Generate a 1D Gaussian kernel with specified standard deviation
std::vector<double> generateGaussianKernel(double sigma) {
    // Determine the kernel size based on sigma
    // A common rule is to use a kernel size of 6*sigma (rounded to next odd number)
    int kernelSize = static_cast<int>(std::ceil(sigma * 6));
    // Ensure kernel size is odd to have a center pixel
    if (kernelSize % 2 == 0) kernelSize++;
    
    std::vector<double> kernel(kernelSize);
    double sum = 0.0;
    int center = kernelSize / 2;
    
    // Fill the kernel with Gaussian values
    for (int i = 0; i < kernelSize; i++) {
        int x = i - center;
        // Gaussian function: G(x) = (1/(sqrt(2π)*σ)) * e^(-(x^2)/(2*σ^2))
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma)) / (std::sqrt(2 * M_PI) * sigma);
        sum += kernel[i];
    }
    
    // Normalize the kernel so it sums to 1
    for (int i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }
    
    return kernel;
}

// Apply Gaussian blur to an image
Image gaussianBlur(const Image& input, double sigma) {
    // Get image dimensions
    int width = input.getWidth();
    int height = input.getHeight();
    int channels = input.getChannels();
    
    // Create a kernel for the Gaussian blur
    std::vector<double> kernel = generateGaussianKernel(sigma);
    int kernelSize = kernel.size();
    int kernelRadius = kernelSize / 2;
    
    // Create temporary image for intermediate result (after horizontal blur)
    Image temp(width, height, channels);
    
    // Create output image
    Image output(width, height, channels);
    
    // Temporary buffers for pixel data
    std::vector<unsigned char> inPixel(channels);
    std::vector<unsigned char> outPixel(channels);
    
    // Horizontal pass (convolve each row with the kernel)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::vector<double> sums(channels, 0.0);
            double weightSum = 0.0;
            
            // Apply kernel horizontally
            for (int i = 0; i < kernelSize; i++) {
                int sampleX = x + (i - kernelRadius);
                
                if (input.getPixel(sampleX, y, inPixel.data())) {
                    double weight = kernel[i];
                    for (int c = 0; c < channels; c++) {
                        sums[c] += inPixel[c] * weight;
                    }
                    weightSum += weight;
                }
            }
            
            // Normalize by total weight used and set pixel
            if (weightSum > 0) {
                for (int c = 0; c < channels; c++) {
                    outPixel[c] = static_cast<unsigned char>(sums[c] / weightSum);
                }
                temp.setPixel(x, y, outPixel.data());
            }
        }
    }
    
    // Vertical pass (convolve each column with the kernel)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::vector<double> sums(channels, 0.0);
            double weightSum = 0.0;
            
            // Apply kernel vertically
            for (int i = 0; i < kernelSize; i++) {
                int sampleY = y + (i - kernelRadius);
                
                if (temp.getPixel(x, sampleY, inPixel.data())) {
                    double weight = kernel[i];
                    for (int c = 0; c < channels; c++) {
                        sums[c] += inPixel[c] * weight;
                    }
                    weightSum += weight;
                }
            }
            
            // Normalize by total weight used and set pixel
            if (weightSum > 0) {
                for (int c = 0; c < channels; c++) {
                    outPixel[c] = static_cast<unsigned char>(sums[c] / weightSum);
                }
                output.setPixel(x, y, outPixel.data());
            }
        }
    }
    
    return output;
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.jpg output.jpg [sigma]" << std::endl;
        std::cerr << "  sigma: Standard deviation for Gaussian blur (default: 2.0)" << std::endl;
        return 1;
    }
    
    // Get input and output filenames
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    
    // Get sigma value (optional)
    double sigma = 2.0; // Default sigma value
    if (argc > 3) {
        try {
            sigma = std::stod(argv[3]);
            if (sigma <= 0) {
                throw std::invalid_argument("Sigma must be positive");
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid sigma value: " << e.what() << std::endl;
            return 1;
        }
    }
    
    // Load input image
    Image inputImage;
    if (!inputImage.load(inputFile)) {
        return 1;
    }
    
    // Apply Gaussian blur
    std::cout << "Applying Gaussian blur with sigma = " << sigma << "..." << std::endl;
    Image blurredImage = gaussianBlur(inputImage, sigma);
    
    // Save output image
    if (!blurredImage.save(outputFile)) {
        return 1;
    }
    
    std::cout << "Gaussian blur applied successfully!" << std::endl;
    return 0;
}