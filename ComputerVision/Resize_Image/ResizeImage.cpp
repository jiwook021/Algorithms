/**
 * @file ResizeImage.cpp
 * @brief Implementation of the seam carving driver
 */

#include "ResizeImage.hpp"
#include <fstream>
#include <iostream>

namespace ResizeImage {

int Run(const std::string& inputPath, const std::string& outputPath,
        int newWidth, int newHeight) {
    std::ifstream fin(inputPath, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Error opening: " << inputPath << std::endl;
        return 1;
    }

    Image img;
    Image_init(&img, fin);
    fin.close();

    if (newWidth <= 0 || newWidth > img.width ||
        newHeight <= 0 || newHeight > img.Height) {
        std::cerr << "Invalid dimensions\n";
        return 1;
    }

    SeamCarve(&img, newWidth, newHeight);

    std::ofstream fout(outputPath);
    Image_print(&img, fout);
    fout.close();
    return 0;
}

}  // namespace ResizeImage
