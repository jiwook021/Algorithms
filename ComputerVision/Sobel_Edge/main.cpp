#include <opencv2/opencv.hpp>  // Main OpenCV header
#include <iostream>

using namespace cv;  // OpenCV namespace
using namespace std;

int main(int argc, char** argv) {
    // Check if image path is provided
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // 1. Read the input image
    // cv::imread - Reads an image from a file
    // Parameters: filename, flags (IMREAD_COLOR loads as BGR)
    Mat inputImage = imread(argv[1], IMREAD_COLOR);
    
    if (inputImage.empty()) {
        cout << "Error: Could not load image" << endl;
        return -1;
    }

    // 2. Convert to grayscale
    // cv::cvtColor - Converts image from one color space to another
    // Parameters: source, destination, color space conversion code
    Mat grayImage;
    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

    // 3. Apply Sobel filter in X and Y directions
    // cv::Sobel - Computes Sobel derivatives
    // Parameters: src, dst, depth(-1=src), dx, dy, ksize
    Mat sobelX, sobelY, sobelCombined;
    Sobel(grayImage, sobelX, -1, 1, 0, 3);  // Horizontal edges
    Sobel(grayImage, sobelY, -1, 0, 1, 3);  // Vertical edges

    // 4. Convert to absolute values and combine
    // cv::convertScaleAbs - Computes absolute values and converts to 8-bit
    Mat absSobelX, absSobelY;
    convertScaleAbs(sobelX, absSobelX);
    convertScaleAbs(sobelY, absSobelY);

    // cv::addWeighted - Calculates weighted sum of two arrays
    // Parameters: src1, alpha, src2, beta, gamma, dst
    addWeighted(absSobelX, 0.5, absSobelY, 0.5, 0, sobelCombined);

    // 5. Apply basic thresholding to enhance edges
    // cv::threshold - Applies fixed-level thresholding
    // Parameters: src, dst, threshold, maxval, type
    Mat thresholdImage;
    threshold(sobelCombined, thresholdImage, 100, 255, THRESH_BINARY);

    // 6. Display results
    // cv::namedWindow - Creates a window
    // cv::imshow - Displays an image in a window
    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("Sobel Edges", WINDOW_NORMAL);
    namedWindow("Threshold Edges", WINDOW_NORMAL);

    imshow("Original Image", inputImage);
    imshow("Sobel Edges", sobelCombined);
    imshow("Threshold Edges", thresholdImage);

    // cv::waitKey - Waits for a key press (0 = infinite)
    waitKey(0);

    // Clean up windows
    // cv::destroyAllWindows - Destroys all created windows
    destroyAllWindows();

    return 0;
}