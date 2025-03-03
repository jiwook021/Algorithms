/**
 * @file main.cpp
 * @brief Driver for GaussianBlur
 */

#include "GaussianBlur.hpp"

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.jpg output.jpg [sigma]" << std::endl;
        std::cerr << "  sigma: Standard deviation for Gaussian blur (default: 2.0)" << std::endl;
        return 1;
    }
    
    // Get input and output filenames
    std::string InputFile = argv[1];
    std::string OutputFile = argv[2];
    
    // Get sigma value (optional)
    double Sigma = 2.0; // Default sigma value
    if (argc > 3) {
        try {
            Sigma = std::stod(argv[3]);
            if (Sigma <= 0) {
                throw std::invalid_argument("Sigma must be positive");
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid sigma value: " << e.what() << std::endl;
            return 1;
        }
    }
    
    // Load input image
    Image InputImage;
    if (!InputImage.Load(InputFile)) {
        return 1;
    }
    
    // Apply Gaussian blur
    std::cout << "Applying Gaussian blur with sigma = " << Sigma << "..." << std::endl;
    Image BlurredImage = gaussianBlur(InputImage, Sigma);
    
    // Save output image
    if (!BlurredImage.Save(OutputFile)) {
        return 1;
    }
    
    std::cout << "Gaussian blur applied successfully!" << std::endl;
    return 0;
}