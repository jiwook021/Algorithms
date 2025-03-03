#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;

// Custom structure to hold RGB values
struct Pixel {
    unsigned char b, g, r;
};

// 1. Custom cvtColor (BGR to Gray)
void myCvtColor(const cv::Mat& src, cv::Mat& dst) {
    dst = cv::Mat(src.rows, src.cols, CV_8UC1);
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // Get BGR values properly from cv::Vec3b
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            // Assign to our Pixel struct
            Pixel p{pixel[0], pixel[1], pixel[2]};  // BGR order
            // Using luminance formula: 0.299R + 0.587G + 0.114B
            dst.at<unsigned char>(y, x) = 
                static_cast<unsigned char>(0.299 * p.r + 0.587 * p.g + 0.114 * p.b);
        }
    }
}

// 2. Custom Sobel filter
void mySobel(const cv::Mat& src, cv::Mat& dstX, cv::Mat& dstY, int kernelSize = 3) {
    dstX = cv::Mat::zeros(src.size(), CV_16S);
    dstY = cv::Mat::zeros(src.size(), CV_16S);
    
    int halfK = kernelSize / 2;
    
    for (int y = halfK; y < src.rows - halfK; y++) {
        for (int x = halfK; x < src.cols - halfK; x++) {
            int gx = 0, gy = 0;
            
            gx = (-1 * src.at<unsigned char>(y-1, x-1)) + (1 * src.at<unsigned char>(y-1, x+1)) +
                 (-2 * src.at<unsigned char>(y, x-1))   + (2 * src.at<unsigned char>(y, x+1)) +
                 (-1 * src.at<unsigned char>(y+1, x-1)) + (1 * src.at<unsigned char>(y+1, x+1));
                 
            gy = (-1 * src.at<unsigned char>(y-1, x-1)) + (-2 * src.at<unsigned char>(y-1, x)) + 
                 (-1 * src.at<unsigned char>(y-1, x+1)) + (1 * src.at<unsigned char>(y+1, x-1)) + 
                 (2 * src.at<unsigned char>(y+1, x))    + (1 * src.at<unsigned char>(y+1, x+1));
                 
            dstX.at<short>(y, x) = gx;
            dstY.at<short>(y, x) = gy;
        }
    }
}

// 3. Custom convertScaleAbs
void myConvertScaleAbs(const cv::Mat& src, cv::Mat& dst) {
    dst = cv::Mat(src.size(), CV_8UC1);
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            dst.at<unsigned char>(y, x) = 
                static_cast<unsigned char>(std::abs(src.at<short>(y, x)));
        }
    }
}

// 4. Custom addWeighted
void myAddWeighted(const cv::Mat& src1, double alpha, 
                  const cv::Mat& src2, double beta, 
                  double gamma, cv::Mat& dst) {
    dst = cv::Mat(src1.size(), CV_8UC1);
    
    for (int y = 0; y < src1.rows; y++) {
        for (int x = 0; x < src1.cols; x++) {
            double value = alpha * src1.at<unsigned char>(y, x) + 
                          beta * src2.at<unsigned char>(y, x) + gamma;
            value = max(0.0, min(255.0, value));
            dst.at<unsigned char>(y, x) = static_cast<unsigned char>(value);
        }
    }
}

// 5. Custom threshold
void myThreshold(const cv::Mat& src, cv::Mat& dst, 
                double thresh, double maxval, int type) {
    dst = cv::Mat(src.size(), CV_8UC1);
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            unsigned char val = src.at<unsigned char>(y, x);
            if (type == 0) {  // THRESH_BINARY
                dst.at<unsigned char>(y, x) = (val > thresh) ? maxval : 0;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        cout << "Error: Could not load image" << endl;
        return -1;
    }

    cv::Mat grayImage;
    myCvtColor(inputImage, grayImage);

    cv::Mat sobelX, sobelY;
    mySobel(grayImage, sobelX, sobelY);

    cv::Mat absSobelX, absSobelY;
    myConvertScaleAbs(sobelX, absSobelX);
    myConvertScaleAbs(sobelY, absSobelY);

    cv::Mat sobelCombined;
    myAddWeighted(absSobelX, 0.5, absSobelY, 0.5, 0.0, sobelCombined);

    cv::Mat thresholdImage;
    myThreshold(sobelCombined, thresholdImage, 100, 255, 0);

    cv::namedWindow("Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Sobel Edges", cv::WINDOW_NORMAL);
    cv::namedWindow("Threshold", cv::WINDOW_NORMAL);

    cv::imshow("Original", inputImage);
    cv::imshow("Sobel Edges", sobelCombined);
    cv::imshow("Threshold", thresholdImage);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}