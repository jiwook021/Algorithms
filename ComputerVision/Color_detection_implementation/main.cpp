#include <opencv2/opencv.hpp>  // Still needed for basic image handling
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Custom inRange: Creates a binary mask for pixels within a color range
void myInRange(const Mat& src, Scalar lower, Scalar upper, Mat& dst) {
    dst = Mat(src.size(), CV_8UC1);  // Single channel output
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3b pixel = src.at<Vec3b>(y, x);  // HSV pixel
            // Check if pixel is within range (H, S, V)
            bool inRange = (pixel[0] >= lower[0] && pixel[0] <= upper[0] &&
                           pixel[1] >= lower[1] && pixel[1] <= upper[1] &&
                           pixel[2] >= lower[2] && pixel[2] <= upper[2]);
            dst.at<unsigned char>(y, x) = inRange ? 255 : 0;
        }
    }
}

// Custom getStructuringElement: Creates a simple rectangular kernel
Mat myGetStructuringElement(int shape, Size ksize) {
    Mat kernel = Mat(ksize, CV_8UC1, Scalar(1));  // All ones
    return kernel;  // For simplicity, assuming MORPH_RECT only
}

// Custom morphologyEx (OPEN operation): Erosion followed by dilation
void myMorphologyEx(const Mat& src, Mat& dst, int op, const Mat& kernel) {
    dst = src.clone();
    if (op != MORPH_OPEN) return;  // Only implementing OPEN for now
    
    // Temporary matrix for erosion
    Mat temp = src.clone();
    int kHalfX = kernel.cols / 2;
    int kHalfY = kernel.rows / 2;
    
    // Erosion: Minimum value in kernel neighborhood
    for (int y = kHalfY; y < src.rows - kHalfY; y++) {
        for (int x = kHalfX; x < src.cols - kHalfX; x++) {
            unsigned char minVal = 255;
            for (int ky = -kHalfY; ky <= kHalfY; ky++) {
                for (int kx = -kHalfX; kx <= kHalfX; kx++) {
                    if (kernel.at<unsigned char>(ky + kHalfY, kx + kHalfX)) {
                        minVal = min(minVal, src.at<unsigned char>(y + ky, x + kx));
                    }
                }
            }
            temp.at<unsigned char>(y, x) = minVal;
        }
    }
    
    // Dilation: Maximum value in kernel neighborhood
    for (int y = kHalfY; y < src.rows - kHalfY; y++) {
        for (int x = kHalfX; x < src.cols - kHalfX; x++) {
            unsigned char maxVal = 0;
            for (int ky = -kHalfY; ky <= kHalfY; ky++) {
                for (int kx = -kHalfX; kx <= kHalfX; kx++) {
                    if (kernel.at<unsigned char>(ky + kHalfY, kx + kHalfX)) {
                        maxVal = max(maxVal, temp.at<unsigned char>(y + ky, x + kx));
                    }
                }
            }
            dst.at<unsigned char>(y, x) = maxVal;
        }
    }
}

// Custom findContours: Simple contour detection (external only)
void myFindContours(const Mat& src, vector<vector<Point>>& contours) {
    contours.clear();
    Mat visited = Mat::zeros(src.size(), CV_8UC1);
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (src.at<unsigned char>(y, x) == 255 && !visited.at<unsigned char>(y, x)) {
                vector<Point> contour;
                vector<Point> stack;
                stack.push_back(Point(x, y));
                
                while (!stack.empty()) {
                    Point p = stack.back();
                    stack.pop_back();
                    
                    if (visited.at<unsigned char>(p.y, p.x)) continue;
                    visited.at<unsigned char>(p.y, p.x) = 255;
                    contour.push_back(p);
                    
                    // Check 4-connected neighbors
                    int dirs[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};
                    for (auto& d : dirs) {
                        int ny = p.y + d[0];
                        int nx = p.x + d[1];
                        if (ny >= 0 && ny < src.rows && nx >= 0 && nx < src.cols &&
                            src.at<unsigned char>(ny, nx) == 255 &&
                            !visited.at<unsigned char>(ny, nx)) {
                            stack.push_back(Point(nx, ny));
                        }
                    }
                }
                if (contour.size() > 10) {  // Filter small contours
                    contours.push_back(contour);
                }
            }
        }
    }
}

// Custom boundingRect: Calculates bounding rectangle for a contour
Rect myBoundingRect(const vector<Point>& contour) {
    int minX = contour[0].x, maxX = contour[0].x;
    int minY = contour[0].y, maxY = contour[0].y;
    
    for (const Point& p : contour) {
        minX = min(minX, p.x);
        maxX = max(maxX, p.x);
        minY = min(minY, p.y);
        maxY = max(maxY, p.y);
    }
    
    return Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    Mat image = imread(argv[1], IMREAD_COLOR);
    if (image.empty()) {
        cout << "Error: Could not load image" << endl;
        return -1;
    }

    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    Mat mask;
    Scalar lowerRed(0, 120, 70);
    Scalar upperRed(10, 255, 255);
    myInRange(hsvImage, lowerRed, upperRed, mask);

    Mat kernel = myGetStructuringElement(MORPH_RECT, Size(5, 5));
    myMorphologyEx(mask, mask, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    myFindContours(mask, contours);

    Mat result = image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        Rect boundingBox = myBoundingRect(contours[i]);
        
        if (boundingBox.width > 20 && boundingBox.height > 20) {
            rectangle(result, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 0), 2);
            string label = "Object " + to_string(i);
            putText(result, label, boundingBox.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }
    }

    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("Mask", WINDOW_NORMAL);
    namedWindow("Detected Objects", WINDOW_NORMAL);

    imshow("Original Image", image);
    imshow("Mask", mask);
    imshow("Detected Objects", result);

    waitKey(0);
    destroyAllWindows();

    return 0;
}