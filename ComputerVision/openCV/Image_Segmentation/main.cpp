#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load an image
    cv::Mat src = cv::imread("input.jpg");
    if (src.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    
    // Apply thresholding to create a binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Create a copy for drawing
    cv::Mat drawing = src.clone();
    
    // Process each contour
    for (size_t i = 0; i < contours.size(); i++) {
        // Calculate area of the contour
        double area = cv::contourArea(contours[i]);
        
        // Filter out small contours (noise)
        if (area < 500) continue;
        
        // Get bounding rectangle
        cv::Rect boundRect = cv::boundingRect(contours[i]);
        
        // Get rotated rectangle
        cv::RotatedRect rotatedRect = cv::minAreaRect(contours[i]);
        cv::Point2f vertices[4];
        rotatedRect.points(vertices);
        
        // Draw contour
        cv::drawContours(drawing, contours, (int)i, cv::Scalar(0, 255, 0), 2);
        
        // Draw bounding rectangle
        cv::rectangle(drawing, boundRect, cv::Scalar(255, 0, 0), 2);
        
        // Draw rotated rectangle
        for (int j = 0; j < 4; j++) {
            cv::line(drawing, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 0, 255), 2);
        }
        
        // Draw center of the contour
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 != 0) {
            cv::Point center(m.m10 / m.m00, m.m01 / m.m00);
            cv::circle(drawing, center, 5, cv::Scalar(255, 0, 255), -1);
            
            // Label with area
            std::string areaText = "A=" + std::to_string(int(area));
            cv::putText(drawing, areaText, center, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                       cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // Show the results
    cv::imshow("Original Image", src);
    cv::imshow("Binary Image", binary);
    cv::imshow("Contours", drawing);
    
    // Print the number of detected objects
    std::cout << "Number of detected objects: " << contours.size() << std::endl;
    
    cv::waitKey(0);
    return 0;
}