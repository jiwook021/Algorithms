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
    Mat image = imread(argv[1], IMREAD_COLOR);
    if (image.empty()) {
        cout << "Error: Could not load image" << endl;
        return -1;
    }

    // 2. Convert to HSV color space (better for color-based detection)
    // cv::cvtColor - Converts image from one color space to another
    // Parameters: source, destination, color space conversion code
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // 3. Define range for red color and create mask
    // cv::inRange - Creates a binary mask where pixels within range are white
    // Parameters: source, lower bound, upper bound, destination
    Mat mask;
    Scalar lowerRed(0, 120, 70);    // Lower HSV range for red
    Scalar upperRed(10, 255, 255);  // Upper HSV range for red
    inRange(hsvImage, lowerRed, upperRed, mask);

    // 4. Apply morphological operation to clean up the mask
    // cv::morphologyEx - Performs advanced morphological transformations
    // Parameters: src, dst, operation, kernel
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);  // Removes small noise

    // 5. Find contours in the mask
    // cv::findContours - Finds contours in a binary image
    // Parameters: image, contours vector, hierarchy, mode, method
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 6. Draw bounding boxes around detected objects
    Mat result = image.clone();  // Copy of original for drawing
    for (size_t i = 0; i < contours.size(); i++) {
        // cv::boundingRect - Calculates the bounding rectangle for a contour
        // Parameters: contour points
        Rect boundingBox = boundingRect(contours[i]);

        // Filter small contours (noise)
        if (boundingBox.width > 20 && boundingBox.height > 20) {
            // cv::rectangle - Draws a rectangle on the image
            // Parameters: image, top-left, bottom-right, color, thickness
            rectangle(result, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 0), 2);
            
            // Optional: Add label
            // cv::putText - Draws text on the image
            // Parameters: image, text, position, font, scale, color, thickness
            string label = "Object " + to_string(i);
            putText(result, label, boundingBox.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }
    }

    // 7. Display results
    // cv::namedWindow - Creates a window
    // cv::imshow - Displays an image in a window
    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("Mask", WINDOW_NORMAL);
    namedWindow("Detected Objects", WINDOW_NORMAL);

    imshow("Original Image", image);
    imshow("Mask", mask);
    imshow("Detected Objects", result);

    // cv::waitKey - Waits for a key press (0 = infinite)
    waitKey(0);

    // cv::destroyAllWindows - Destroys all created windows
    destroyAllWindows();

    return 0;
}