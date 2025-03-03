/**
 * @file main.cpp
 * @brief Driver for DownSample module
 */

#include "DownSample.hpp"

int main(int argc, char** argv) {
    std::string inputFile  = (argc > 1) ? argv[1] : "input.png";
    std::string outputFile = (argc > 2) ? argv[2] : "output.png";

    try {
        DownSample::Image img(inputFile);
        std::cout << "Loaded " << img.GetWidth() << "x" << img.GetHeight() << std::endl;

        auto downsampled = img.HalfSize();
        if (downsampled->Save(outputFile))
            std::cout << "Saved " << outputFile << " ("
                      << downsampled->GetWidth() << "x"
                      << downsampled->GetHeight() << ")\n";
        else
            std::cerr << "Failed to save image\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
