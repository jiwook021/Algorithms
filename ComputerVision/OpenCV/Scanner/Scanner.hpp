/**
 * @file Scanner.hpp
 * @brief Document scanner with perspective correction
 * @details Detects document edges using Canny + Hough or contour detection, computes the perspective transform, and applies warpPerspective to produce a flat scan. Includes contrast enhancement post-processing.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// Function to order points in clockwise order
inline std::vector<cv::Point2f> OrderPoints(const std::vector<cv::Point2f>& Pts) {
    std::vector<cv::Point2f> Ordered(4);
    
    // Sum and difference of coordinates
    std::vector<float> Sum(4), Diff(4);
    for (int i = 0; i < 4; i++) {
        Sum[i] = Pts[i].x + Pts[i].y;
        Diff[i] = Pts[i].x - Pts[i].y;
    }
    
    // Top-left: smallest sum
    Ordered[0] = Pts[std::min_element(Sum.begin(), Sum.end()) - Sum.begin()];
    // Bottom-right: largest sum
    Ordered[2] = Pts[std::max_element(Sum.begin(), Sum.end()) - Sum.begin()];
    // Top-right: smallest difference
    Ordered[1] = Pts[std::min_element(Diff.begin(), Diff.end()) - Diff.begin()];
    // Bottom-left: largest difference
    Ordered[3] = Pts[std::max_element(Diff.begin(), Diff.end()) - Diff.begin()];
    
    return Ordered;
}

// Function to perform perspective transform
inline cv::Mat FourPointTransform(const cv::Mat& Image, const std::vector<cv::Point2f>& Pts) {
    // Get ordered points
    std::vector<cv::Point2f> Rect = OrderPoints(Pts);
    cv::Point2f Tl = Rect[0], Tr = Rect[1], Br = Rect[2], Bl = Rect[3];
    
    // Compute width of the new image
    float WidthA = std::sqrt(std::pow(Br.x - Bl.x, 2) + std::pow(Br.y - Bl.y, 2));
    float WidthB = std::sqrt(std::pow(Tr.x - Tl.x, 2) + std::pow(Tr.y - Tl.y, 2));
    int MaxWidth = std::max(int(WidthA), int(WidthB));
    
    // Compute height of the new image
    float HeightA = std::sqrt(std::pow(Tr.x - Br.x, 2) + std::pow(Tr.y - Br.y, 2));
    float HeightB = std::sqrt(std::pow(Tl.x - Bl.x, 2) + std::pow(Tl.y - Bl.y, 2));
    int MaxHeight = std::max(int(HeightA), int(HeightB));
    
    // Define the destination points
    std::vector<cv::Point2f> Dst = {
        cv::Point2f(0, 0),
        cv::Point2f(MaxWidth - 1, 0),
        cv::Point2f(MaxWidth - 1, MaxHeight - 1),
        cv::Point2f(0, MaxHeight - 1)
    };
    
    // Compute the perspective transform matrix and apply it
    cv::Mat M = cv::getPerspectiveTransform(Rect, Dst);
    cv::Mat Warped;
    cv::warpPerspective(Image, Warped, M, cv::Size(MaxWidth, MaxHeight));
    
    return Warped;
}

// Function to enhance document readability
inline cv::Mat EnhanceDocument(const cv::Mat& Image) {
    cv::Mat Enhanced;
    
    // Convert to grayscale if not already
    if (Image.channels() == 3) {
        cv::cvtColor(Image, Enhanced, cv::COLOR_BGR2GRAY);
    } else {
        Enhanced = Image.clone();
    }
    
    // Apply bilateral filtering to smooth while preserving edges
    cv::Mat Bilateral;
    cv::bilateralFilter(Enhanced, Bilateral, 9, 75, 75);
    
    // Apply adaptive thresholding to binarize the image
    cv::Mat Thresh;
    cv::adaptiveThreshold(Bilateral, Thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY, 11, 2);
    
    // Apply morphological operations to remove noise
    cv::Mat Kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat Morphology;
    cv::morphologyEx(Thresh, Morphology, cv::MORPH_CLOSE, Kernel);
    
    return Morphology;
}

// Function to detect if document is upside down using text orientation
inline bool IsUpsideDown(const cv::Mat& Image) {
    // This is a simplified version - in a real app, you'd use OCR or more sophisticated methods
    // For this example, we'll just detect if there are more dark pixels in the top half than bottom
    
    cv::Mat Gray;
    if (Image.channels() == 3) {
        cv::cvtColor(Image, Gray, cv::COLOR_BGR2GRAY);
    } else {
        Gray = Image.clone();
    }
    
    // Threshold to get binary image
    cv::Mat Binary;
    cv::threshold(Gray, Binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    
    // Count dark pixels in top and bottom halves
    int Height = Binary.rows;
    cv::Mat TopHalf = Binary(cv::Rect(0, 0, Binary.cols, Height / 2));
    cv::Mat BottomHalf = Binary(cv::Rect(0, Height / 2, Binary.cols, Height / 2));
    
    int TopPixels = cv::countNonZero(TopHalf);
    int BottomPixels = cv::countNonZero(BottomHalf);
    
    // If more dark pixels at the top than bottom, document may be upside down
    return TopPixels > BottomPixels * 1.2;
}

// Function to auto-rotate to correct orientation
inline cv::Mat CorrectOrientation(const cv::Mat& Image) {
    if (IsUpsideDown(Image)) {
        cv::Mat Rotated;
        cv::rotate(Image, Rotated, cv::ROTATE_180);
        return Rotated;
    }
    return Image.clone();
}

// Function to process and enhance document shadows
inline cv::Mat RemoveShadows(const cv::Mat& Image) {
    cv::Mat Bgr;
    if (Image.channels() == 1) {
        cv::cvtColor(Image, Bgr, cv::COLOR_GRAY2BGR);
    } else {
        Bgr = Image.clone();
    }
    
    // Convert to Lab color space
    cv::Mat Lab;
    cv::cvtColor(Bgr, Lab, cv::COLOR_BGR2Lab);
    
    // Split the Lab channels
    std::vector<cv::Mat> LabPlanes(3);
    cv::split(Lab, LabPlanes);
    
    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    cv::Ptr<cv::CLAHE> Clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat Dst;
    Clahe->apply(LabPlanes[0], Dst);
    
    // Merge back the Lab channels
    LabPlanes[0] = Dst;
    cv::merge(LabPlanes, Lab);
    
    // Convert back to BGR
    cv::Mat Result;
    cv::cvtColor(Lab, Result, cv::COLOR_Lab2BGR);
    
    return Result;
}

// Interactive document scanner class

class DocumentScanner {
private:
    cv::Mat OriginalImage;
    cv::Mat ProcessedImage;
    cv::Mat DisplayImage;
    std::vector<cv::Point2f> DocumentCorners;
    bool CornersSelected;
    std::string WindowName;
    int OutputMaxDimension = 0;
    bool OutputMaintainAspectRatio = true;
    int JpegQuality = 85;  // Default to a more reasonable quality
    
    // Mouse callback function helper
    static void MouseCallbackHelper(int Event, int x, int y, int Flags, void* Userdata) {
        DocumentScanner* Scanner = static_cast<DocumentScanner*>(Userdata);
        Scanner->MouseCallback(Event, x, y, Flags);
    }
    
    // Actual mouse callback implementation
    void MouseCallback(int Event, int x, int y, int Flags) {
        if (Event == cv::EVENT_LBUTTONDOWN) {
            // If we have less than 4 corners, add a new one
            if (DocumentCorners.size() < 4) {
                DocumentCorners.push_back(cv::Point2f(x, y));
                
                // Draw the point
                cv::circle(DisplayImage, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
                
                // If we now have 4 corners, draw the document outline
                if (DocumentCorners.size() == 4) {
                    for (int i = 0; i < 4; i++) {
                        cv::line(DisplayImage, DocumentCorners[i], DocumentCorners[(i + 1) % 4],
                               cv::Scalar(0, 255, 0), 2);
                    }
                    CornersSelected = true;
                }
                
                cv::imshow(WindowName, DisplayImage);
            }
        }
    }
    
public:
    DocumentScanner(const std::string& ImagePath) : CornersSelected(false), WindowName("Document Scanner") {
        // Load the image
        OriginalImage = cv::imread(ImagePath);
        if (OriginalImage.empty()) {
            throw std::runtime_error("Could not open or find the image: " + ImagePath);
        }
        
        // Resize for display if too large
        if (OriginalImage.rows > 800) {
            double Scale = 800.0 / OriginalImage.rows;
            cv::resize(OriginalImage, OriginalImage, cv::Size(), Scale, Scale);
        }
        
        ProcessedImage = OriginalImage.clone();
        DisplayImage = OriginalImage.clone();
    }
    
    // Set output size constraints
    void SetOutputSize(int MaxDimension = 0, bool MaintainAspectRatio = true) {
        OutputMaxDimension = MaxDimension;
        OutputMaintainAspectRatio = MaintainAspectRatio;
    }
    
    // Set JPEG quality
    void SetJpegQuality(int Quality) {
        JpegQuality = std::max(0, std::min(100, Quality));
    }
    
    // Auto detect document corners
    bool AutoDetectCorners() {
        // Convert to grayscale and apply edge detection
        cv::Mat Gray, Blurred, Edged;
        cv::cvtColor(OriginalImage, Gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(Gray, Blurred, cv::Size(5, 5), 0);
        cv::Canny(Blurred, Edged, 75, 200);
        
        // Find contours
        std::vector<std::vector<cv::Point>> Contours;
        cv::findContours(Edged.clone(), Contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        
        // Sort by area (descending)
        std::sort(Contours.begin(), Contours.end(), 
                 [](const std::vector<cv::Point>& C1, const std::vector<cv::Point>& C2) {
                     return cv::contourArea(C1) > cv::contourArea(C2);
                 });
        
        // Look for document contour
        for (const auto& Contour : Contours) {
            double Peri = cv::arcLength(Contour, true);
            std::vector<cv::Point> Approx;
            cv::approxPolyDP(Contour, Approx, 0.02 * Peri, true);
            
            if (Approx.size() == 4 && cv::contourArea(Approx) > 1000) {
                DocumentCorners.clear();
                for (const auto& p : Approx) {
                    DocumentCorners.push_back(cv::Point2f(p.x, p.y));
                }
                
                // Draw corners and outline
                DisplayImage = OriginalImage.clone();
                for (int i = 0; i < 4; i++) {
                    cv::circle(DisplayImage, DocumentCorners[i], 5, cv::Scalar(0, 255, 0), -1);
                    cv::line(DisplayImage, DocumentCorners[i], DocumentCorners[(i + 1) % 4],
                           cv::Scalar(0, 255, 0), 2);
                }
                
                CornersSelected = true;
                return true;
            }
        }
        
        return false;
    }
    
    // Manual corner selection
    void SelectCorners() {
        DisplayImage = OriginalImage.clone();
        
        // Create window and set mouse callback
        cv::namedWindow(WindowName);
        cv::setMouseCallback(WindowName, MouseCallbackHelper, this);
        
        std::cout << "Select the 4 corners of the document (clockwise from top-left)" << std::endl;
        
        while (!CornersSelected || cv::waitKey(1) != 13) { // Enter key to confirm
            cv::imshow(WindowName, DisplayImage);
        }
        
        cv::destroyWindow(WindowName);
    }
    
    // Process the document and return the result
    cv::Mat ProcessDocument(bool EnhanceText = true, bool CorrectRotation = true, bool RemoveShadowsFlag = true, bool PreserveColor = true) {
        if (!CornersSelected || DocumentCorners.size() != 4) {
            throw std::runtime_error("Document corners not selected");
        }
        
        // Apply perspective transform
        cv::Mat Warped = FourPointTransform(OriginalImage, DocumentCorners);
        
        // Correct orientation if needed
        if (CorrectRotation) {
            Warped = CorrectOrientation(Warped);
        }
        
        // Remove shadows if requested
        if (RemoveShadowsFlag) {
            Warped = RemoveShadows(Warped);
        }
        
        // For color output with enhanced quality
        if (PreserveColor) {
            // Apply sharpening to improve details while keeping color
            cv::Mat Blurred, Sharpened;
            cv::GaussianBlur(Warped, Blurred, cv::Size(0, 0), 3);
            cv::addWeighted(Warped, 1.5, Blurred, -0.5, 0, Sharpened);
            
            // Apply mild color correction to enhance vibrancy
            cv::Mat ColorEnhanced;
            cv::convertScaleAbs(Sharpened, ColorEnhanced, 1.1, 5);
            
            // Use bilateral filter instead of detailEnhance (which requires opencv_photo)
            cv::Mat Enhanced;
            cv::bilateralFilter(ColorEnhanced, Enhanced, 5, 50, 50);
            
            ProcessedImage = Enhanced;
        }
        // Enhance text readability if requested (black and white output)
        else if (EnhanceText) {
            ProcessedImage = EnhanceDocument(Warped);
        } else {
            ProcessedImage = Warped;
        }
        
        // Apply size constraints if specified
        if (OutputMaxDimension > 0) {
            int width = ProcessedImage.cols;
            int Height = ProcessedImage.rows;
            
            // Calculate scaling factor
            double Scale = 1.0;
            if (width > Height) {
                Scale = (double)OutputMaxDimension / width;
            } else {
                Scale = (double)OutputMaxDimension / Height;
            }
            
            // Only scale down, never up
            if (Scale < 1.0) {
                // Resize the image with area interpolation for downsampling (better quality)
                cv::resize(ProcessedImage, ProcessedImage, cv::Size(), Scale, Scale, cv::INTER_AREA);
            }
        }
        
        return ProcessedImage;
    }
    
    // Save the processed document
    void SaveDocument(const std::string& OutputPath) {
        if (ProcessedImage.empty()) {
            throw std::runtime_error("No processed document to save");
        }
        
        // Create parameters for JPEG output with controlled quality
        std::vector<int> CompressionParams;
        CompressionParams.push_back(cv::IMWRITE_JPEG_QUALITY);
        CompressionParams.push_back(JpegQuality);
        
        cv::imwrite(OutputPath, ProcessedImage, CompressionParams);
        std::cout << "Document saved to: " << OutputPath << " with quality " << JpegQuality << std::endl;
    }
    
    // Show results
    void ShowResults() {
        if (ProcessedImage.empty()) {
            throw std::runtime_error("No processed document to display");
        }
        
        cv::imshow("Original Image", OriginalImage);
        cv::imshow("Processed Document", ProcessedImage);
        cv::waitKey(0);
    }
};

