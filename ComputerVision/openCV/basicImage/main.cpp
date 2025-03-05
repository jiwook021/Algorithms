#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Read an image from file
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    // Image properties
    std::cout << "Image dimensions: " << image.cols << "x" << image.rows << std::endl;
    std::cout << "Number of channels: " << image.channels() << std::endl;
    
    // Accessing pixel values
    cv::Vec3b pixel = image.at<cv::Vec3b>(100, 100); // Pixel at (100,100)
    std::cout << "Pixel at (100,100): B=" << (int)pixel[0] 
              << ", G=" << (int)pixel[1] 
              << ", R=" << (int)pixel[2] << std::endl;
    
    // Modifying a pixel
    image.at<cv::Vec3b>(100, 100) = cv::Vec3b(0, 0, 255); // Set to red
    
    // Creating a region of interest (ROI)
    cv::Rect roi(100, 100, 200, 200); // x, y, width, height
    cv::Mat imageROI = image(roi);
    
    // Converting color space
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    // Saving images
    cv::imwrite("output_gray.jpg", grayImage);
    cv::imwrite("output_roi.jpg", imageROI);
    
    // Display the images
    cv::imshow("Original Image", image);
    cv::imshow("Gray Image", grayImage);
    cv::imshow("ROI", imageROI);
    
    cv::waitKey(0);
    return 0;
}