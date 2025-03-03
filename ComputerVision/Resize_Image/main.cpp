/**
 * @file main.cpp
 * @brief Driver for ResizeImage (seam carving)
 */

#include "ResizeImage.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cout << "Usage: " << argv[0] << " IN_FILE OUT_FILE WIDTH [HEIGHT]\n";
        return 1;
    }

    int newWidth  = std::atoi(argv[3]);
    int newHeight = (argc == 5) ? std::atoi(argv[4]) : newWidth;

    return ResizeImage::Run(argv[1], argv[2], newWidth, newHeight);
}
