// Force the old ABI for this translation unit only
#define _GLIBCXX_USE_CXX11_ABI 0

// Include our wrapper header first
#include "opencv_wrapper.h"

// Then include OpenCV
#include <opencv2/opencv.hpp>

// Define our constants with different names to avoid conflicts
const int WINDOW_NORMAL_CONST = cv::WINDOW_NORMAL;
const int WINDOW_AUTOSIZE_CONST = cv::WINDOW_AUTOSIZE;
const int FONT_HERSHEY_SIMPLEX_CONST = cv::FONT_HERSHEY_SIMPLEX;
const int LINE_8_CONST = cv::LINE_8;
const int COLOR_BGR2RGB_CONST = cv::COLOR_BGR2RGB;
const int CV_32F_CONST = CV_32F;  // CV_32F is a macro, not a cv:: namespace constant

// Image I/O operations
cv::Mat* load_image(const std::string& filename, int flags) {
    // Create a new Mat on the heap and return its pointer
    cv::Mat* result = new cv::Mat(cv::imread(filename, flags));
    return result;
}

bool save_image(const std::string& filename, cv::Mat* img) {
    return cv::imwrite(filename, *img);
}

// Window management
void create_window(const std::string& winname, int flags) {
    cv::namedWindow(winname, flags);
}

void show_image(const std::string& winname, cv::Mat* img) {
    cv::imshow(winname, *img);
}

int wait_key(int delay) {
    return cv::waitKey(delay);
}

void destroy_window(const std::string& winname) {
    cv::destroyWindow(winname);
}

void destroy_all_windows() {
    cv::destroyAllWindows();
}

// Drawing operations
void draw_text(cv::Mat* img, const std::string& text, cv::Point org, 
               int fontFace, double fontScale, cv::Scalar color, 
               int thickness, int lineType, bool bottomLeftOrigin) {
    cv::putText(*img, text, org, fontFace, fontScale, color, 
                thickness, lineType, bottomLeftOrigin);
}

void draw_rectangle(cv::Mat* img, cv::Rect rect, cv::Scalar color, 
                    int thickness, int lineType, int shift) {
    cv::rectangle(*img, rect, color, thickness, lineType, shift);
}

// Image processing
void resize_image(const cv::Mat* src, cv::Mat* dst, cv::Size size, 
                  double fx, double fy, int interpolation) {
    cv::resize(*src, *dst, size, fx, fy, interpolation);
}

void cvt_color(const cv::Mat* src, cv::Mat* dst, int code) {
    cv::cvtColor(*src, *dst, code);
}

void subtract_scalar(cv::Mat* img, const cv::Scalar& val) {
    cv::subtract(*img, val, *img);
}

void divide_scalar(cv::Mat* img, const cv::Scalar& val) {
    cv::divide(*img, val, *img);
}