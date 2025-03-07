#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// Function to order points in clockwise order
std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> ordered(4);
    
    // Sum and difference of coordinates
    std::vector<float> sum(4), diff(4);
    for (int i = 0; i < 4; i++) {
        sum[i] = pts[i].x + pts[i].y;
        diff[i] = pts[i].x - pts[i].y;
    }
    
    // Top-left: smallest sum
    ordered[0] = pts[std::min_element(sum.begin(), sum.end()) - sum.begin()];
    // Bottom-right: largest sum
    ordered[2] = pts[std::max_element(sum.begin(), sum.end()) - sum.begin()];
    // Top-right: smallest difference
    ordered[1] = pts[std::min_element(diff.begin(), diff.end()) - diff.begin()];
    // Bottom-left: largest difference
    ordered[3] = pts[std::max_element(diff.begin(), diff.end()) - diff.begin()];
    
    return ordered;
}

// Function to perform perspective transform
cv::Mat fourPointTransform(const cv::Mat& image, const std::vector<cv::Point2f>& pts) {
    // Get ordered points
    std::vector<cv::Point2f> rect = orderPoints(pts);
    cv::Point2f tl = rect[0], tr = rect[1], br = rect[2], bl = rect[3];
    
    // Compute width of the new image
    float widthA = std::sqrt(std::pow(br.x - bl.x, 2) + std::pow(br.y - bl.y, 2));
    float widthB = std::sqrt(std::pow(tr.x - tl.x, 2) + std::pow(tr.y - tl.y, 2));
    int maxWidth = std::max(int(widthA), int(widthB));
    
    // Compute height of the new image
    float heightA = std::sqrt(std::pow(tr.x - br.x, 2) + std::pow(tr.y - br.y, 2));
    float heightB = std::sqrt(std::pow(tl.x - bl.x, 2) + std::pow(tl.y - bl.y, 2));
    int maxHeight = std::max(int(heightA), int(heightB));
    
    // Define the destination points
    std::vector<cv::Point2f> dst = {
        cv::Point2f(0, 0),
        cv::Point2f(maxWidth - 1, 0),
        cv::Point2f(maxWidth - 1, maxHeight - 1),
        cv::Point2f(0, maxHeight - 1)
    };
    
    // Compute the perspective transform matrix and apply it
    cv::Mat M = cv::getPerspectiveTransform(rect, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(maxWidth, maxHeight));
    
    return warped;
}

// Function to enhance document readability
cv::Mat enhanceDocument(const cv::Mat& image) {
    cv::Mat enhanced;
    
    // Convert to grayscale if not already
    if (image.channels() == 3) {
        cv::cvtColor(image, enhanced, cv::COLOR_BGR2GRAY);
    } else {
        enhanced = image.clone();
    }
    
    // Apply bilateral filtering to smooth while preserving edges
    cv::Mat bilateral;
    cv::bilateralFilter(enhanced, bilateral, 9, 75, 75);
    
    // Apply adaptive thresholding to binarize the image
    cv::Mat thresh;
    cv::adaptiveThreshold(bilateral, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY, 11, 2);
    
    // Apply morphological operations to remove noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat morphology;
    cv::morphologyEx(thresh, morphology, cv::MORPH_CLOSE, kernel);
    
    return morphology;
}

// Function to detect if document is upside down using text orientation
bool isUpsideDown(const cv::Mat& image) {
    // This is a simplified version - in a real app, you'd use OCR or more sophisticated methods
    // For this example, we'll just detect if there are more dark pixels in the top half than bottom
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Threshold to get binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    
    // Count dark pixels in top and bottom halves
    int height = binary.rows;
    cv::Mat topHalf = binary(cv::Rect(0, 0, binary.cols, height / 2));
    cv::Mat bottomHalf = binary(cv::Rect(0, height / 2, binary.cols, height / 2));
    
    int topPixels = cv::countNonZero(topHalf);
    int bottomPixels = cv::countNonZero(bottomHalf);
    
    // If more dark pixels at the top than bottom, document may be upside down
    return topPixels > bottomPixels * 1.2;
}

// Function to auto-rotate to correct orientation
cv::Mat correctOrientation(const cv::Mat& image) {
    if (isUpsideDown(image)) {
        cv::Mat rotated;
        cv::rotate(image, rotated, cv::ROTATE_180);
        return rotated;
    }
    return image.clone();
}

// Function to process and enhance document shadows
cv::Mat removeShadows(const cv::Mat& image) {
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image.clone();
    }
    
    // Convert to Lab color space
    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    
    // Split the Lab channels
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab, lab_planes);
    
    // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat dst;
    clahe->apply(lab_planes[0], dst);
    
    // Merge back the Lab channels
    lab_planes[0] = dst;
    cv::merge(lab_planes, lab);
    
    // Convert back to BGR
    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    
    return result;
}

// Interactive document scanner class
class DocumentScanner {
private:
    cv::Mat originalImage;
    cv::Mat processedImage;
    cv::Mat displayImage;
    std::vector<cv::Point2f> documentCorners;
    bool cornersSelected;
    std::string windowName;
    int outputMaxDimension = 0;
    bool outputMaintainAspectRatio = true;
    int jpegQuality = 85;  // Default to a more reasonable quality
    
    // Mouse callback function helper
    static void mouseCallbackHelper(int event, int x, int y, int flags, void* userdata) {
        DocumentScanner* scanner = static_cast<DocumentScanner*>(userdata);
        scanner->mouseCallback(event, x, y, flags);
    }
    
    // Actual mouse callback implementation
    void mouseCallback(int event, int x, int y, int flags) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            // If we have less than 4 corners, add a new one
            if (documentCorners.size() < 4) {
                documentCorners.push_back(cv::Point2f(x, y));
                
                // Draw the point
                cv::circle(displayImage, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
                
                // If we now have 4 corners, draw the document outline
                if (documentCorners.size() == 4) {
                    for (int i = 0; i < 4; i++) {
                        cv::line(displayImage, documentCorners[i], documentCorners[(i + 1) % 4],
                               cv::Scalar(0, 255, 0), 2);
                    }
                    cornersSelected = true;
                }
                
                cv::imshow(windowName, displayImage);
            }
        }
    }
    
public:
    DocumentScanner(const std::string& imagePath) : cornersSelected(false), windowName("Document Scanner") {
        // Load the image
        originalImage = cv::imread(imagePath);
        if (originalImage.empty()) {
            throw std::runtime_error("Could not open or find the image: " + imagePath);
        }
        
        // Resize for display if too large
        if (originalImage.rows > 800) {
            double scale = 800.0 / originalImage.rows;
            cv::resize(originalImage, originalImage, cv::Size(), scale, scale);
        }
        
        processedImage = originalImage.clone();
        displayImage = originalImage.clone();
    }
    
    // Set output size constraints
    void setOutputSize(int maxDimension = 0, bool maintainAspectRatio = true) {
        outputMaxDimension = maxDimension;
        outputMaintainAspectRatio = maintainAspectRatio;
    }
    
    // Set JPEG quality
    void setJpegQuality(int quality) {
        jpegQuality = std::max(0, std::min(100, quality));
    }
    
    // Auto detect document corners
    bool autoDetectCorners() {
        // Convert to grayscale and apply edge detection
        cv::Mat gray, blurred, edged;
        cv::cvtColor(originalImage, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::Canny(blurred, edged, 75, 200);
        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edged.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        
        // Sort by area (descending)
        std::sort(contours.begin(), contours.end(), 
                 [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                     return cv::contourArea(c1) > cv::contourArea(c2);
                 });
        
        // Look for document contour
        for (const auto& contour : contours) {
            double peri = cv::arcLength(contour, true);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, 0.02 * peri, true);
            
            if (approx.size() == 4 && cv::contourArea(approx) > 1000) {
                documentCorners.clear();
                for (const auto& p : approx) {
                    documentCorners.push_back(cv::Point2f(p.x, p.y));
                }
                
                // Draw corners and outline
                displayImage = originalImage.clone();
                for (int i = 0; i < 4; i++) {
                    cv::circle(displayImage, documentCorners[i], 5, cv::Scalar(0, 255, 0), -1);
                    cv::line(displayImage, documentCorners[i], documentCorners[(i + 1) % 4],
                           cv::Scalar(0, 255, 0), 2);
                }
                
                cornersSelected = true;
                return true;
            }
        }
        
        return false;
    }
    
    // Manual corner selection
    void selectCorners() {
        displayImage = originalImage.clone();
        
        // Create window and set mouse callback
        cv::namedWindow(windowName);
        cv::setMouseCallback(windowName, mouseCallbackHelper, this);
        
        std::cout << "Select the 4 corners of the document (clockwise from top-left)" << std::endl;
        
        while (!cornersSelected || cv::waitKey(1) != 13) { // Enter key to confirm
            cv::imshow(windowName, displayImage);
        }
        
        cv::destroyWindow(windowName);
    }
    
    // Process the document and return the result
    cv::Mat processDocument(bool enhanceText = true, bool correctRotation = true, bool removeShadowsFlag = true, bool preserveColor = true) {
        if (!cornersSelected || documentCorners.size() != 4) {
            throw std::runtime_error("Document corners not selected");
        }
        
        // Apply perspective transform
        cv::Mat warped = fourPointTransform(originalImage, documentCorners);
        
        // Correct orientation if needed
        if (correctRotation) {
            warped = correctOrientation(warped);
        }
        
        // Remove shadows if requested
        if (removeShadowsFlag) {
            warped = removeShadows(warped);
        }
        
        // For color output with enhanced quality
        if (preserveColor) {
            // Apply sharpening to improve details while keeping color
            cv::Mat blurred, sharpened;
            cv::GaussianBlur(warped, blurred, cv::Size(0, 0), 3);
            cv::addWeighted(warped, 1.5, blurred, -0.5, 0, sharpened);
            
            // Apply mild color correction to enhance vibrancy
            cv::Mat colorEnhanced;
            cv::convertScaleAbs(sharpened, colorEnhanced, 1.1, 5);
            
            // Use bilateral filter instead of detailEnhance (which requires opencv_photo)
            cv::Mat enhanced;
            cv::bilateralFilter(colorEnhanced, enhanced, 5, 50, 50);
            
            processedImage = enhanced;
        }
        // Enhance text readability if requested (black and white output)
        else if (enhanceText) {
            processedImage = enhanceDocument(warped);
        } else {
            processedImage = warped;
        }
        
        // Apply size constraints if specified
        if (outputMaxDimension > 0) {
            int width = processedImage.cols;
            int height = processedImage.rows;
            
            // Calculate scaling factor
            double scale = 1.0;
            if (width > height) {
                scale = (double)outputMaxDimension / width;
            } else {
                scale = (double)outputMaxDimension / height;
            }
            
            // Only scale down, never up
            if (scale < 1.0) {
                // Resize the image with area interpolation for downsampling (better quality)
                cv::resize(processedImage, processedImage, cv::Size(), scale, scale, cv::INTER_AREA);
            }
        }
        
        return processedImage;
    }
    
    // Save the processed document
    void saveDocument(const std::string& outputPath) {
        if (processedImage.empty()) {
            throw std::runtime_error("No processed document to save");
        }
        
        // Create parameters for JPEG output with controlled quality
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(jpegQuality);
        
        cv::imwrite(outputPath, processedImage, compression_params);
        std::cout << "Document saved to: " << outputPath << " with quality " << jpegQuality << std::endl;
    }
    
    // Show results
    void showResults() {
        if (processedImage.empty()) {
            throw std::runtime_error("No processed document to display");
        }
        
        cv::imshow("Original Image", originalImage);
        cv::imshow("Processed Document", processedImage);
        cv::waitKey(0);
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path> [output_path] [max_dimension] [jpeg_quality]" << std::endl;
        return -1;
    }
    
    // Default settings
    std::string outputPath = "scanned_document.jpg";
    int maxDimension = 1200;  // Default max dimension
    int quality = 100;         // Default JPEG quality
    
    // Parse command line arguments
    if (argc >= 3) outputPath = argv[2];
    if (argc >= 4) maxDimension = std::stoi(argv[3]);
    if (argc >= 5) quality = std::stoi(argv[4]);
    
    try {
        // Create document scanner
        DocumentScanner scanner(argv[1]);
        
        // Set output size and quality
        scanner.setOutputSize(maxDimension);
        scanner.setJpegQuality(quality);
        
        // Try auto-detection first
        std::cout << "Attempting automatic document detection..." << std::endl;
        bool autoDetected = scanner.autoDetectCorners();
        
        if (!autoDetected) {
            std::cout << "Automatic detection failed. Please select corners manually." << std::endl;
            scanner.selectCorners();
        } else {
            std::cout << "Document detected automatically!" << std::endl;
        }
        
        // Process the document with all enhancements and preserve color
        scanner.processDocument(true, true, true, true);
        
        // Save the document
        scanner.saveDocument(outputPath);
        
        // Show results
        scanner.showResults();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}