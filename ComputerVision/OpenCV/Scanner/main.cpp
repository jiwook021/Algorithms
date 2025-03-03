/**
 * @file main.cpp
 * @brief Driver for Scanner
 */

#include "Scanner.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path> [output_path] [max_dimension] [jpeg_quality]" << std::endl;
        return -1;
    }
    
    // Default settings
    std::string OutputPath = "scanned_document.jpg";
    int MaxDimension = 1200;  // Default max dimension
    int Quality = 100;         // Default JPEG quality
    
    // Parse command line arguments
    if (argc >= 3) OutputPath = argv[2];
    if (argc >= 4) MaxDimension = std::stoi(argv[3]);
    if (argc >= 5) Quality = std::stoi(argv[4]);
    
    try {
        // Create document scanner
        DocumentScanner Scanner(argv[1]);
        
        // Set output size and quality
        Scanner.SetOutputSize(MaxDimension);
        Scanner.SetJpegQuality(Quality);
        
        // Try auto-detection first
        std::cout << "Attempting automatic document detection..." << std::endl;
        bool AutoDetected = Scanner.AutoDetectCorners();
        
        if (!AutoDetected) {
            std::cout << "Automatic detection failed. Please select corners manually." << std::endl;
            Scanner.SelectCorners();
        } else {
            std::cout << "Document detected automatically!" << std::endl;
        }
        
        // Process the document with all enhancements and preserve color
        Scanner.ProcessDocument(true, true, true, true);
        
        // Save the document
        Scanner.SaveDocument(OutputPath);
        
        // Show results
        Scanner.ShowResults();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}