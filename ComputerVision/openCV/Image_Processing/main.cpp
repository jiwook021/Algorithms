#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load an image
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Resizing an image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(640, 480));
    
    // Blurring/Smoothing
    cv::Mat blurred;
    // Gaussian Blur
    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 0);
    
    // Image thresholding
    cv::Mat thresholded;
    cv::threshold(gray, thresholded, 127, 255, cv::THRESH_BINARY);
    
    // Edge detection using Canny
    cv::Mat edges;
    cv::Canny(gray, edges, 100, 200);
    
    // Image rotation
    cv::Mat rotated;
    cv::Point2f center((float)image.cols/2, (float)image.rows/2);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, 45, 1.0); // 45 degrees
    cv::warpAffine(image, rotated, rotationMatrix, image.size());
    
    // Display results
    cv::imshow("Original", image);
    cv::imshow("Grayscale", gray);
    cv::imshow("Resized", resized);
    cv::imshow("Blurred", blurred);
    cv::imshow("Thresholded", thresholded);
    cv::imshow("Edges", edges);
    cv::imshow("Rotated", rotated);
    
    cv::waitKey(0);
    return 0;
}