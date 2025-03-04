#pragma once

// We're intentionally NOT including OpenCV headers in the header
// to avoid propagating any ABI issues to client code
#include <string>

// Forward declare OpenCV types to avoid including headers
namespace cv {
    class Mat;
    template<typename _Tp> class Point_;
    typedef Point_<int> Point;
    template<typename _Tp> class Scalar_;
    typedef Scalar_<double> Scalar;
    template<typename _Tp> class Size_;
    typedef Size_<int> Size;
    template<typename _Tp> class Rect_;
    typedef Rect_<int> Rect;
}

// Safer forward declarations for constants
extern const int WINDOW_NORMAL_CONST;
extern const int WINDOW_AUTOSIZE_CONST;
extern const int FONT_HERSHEY_SIMPLEX_CONST;
extern const int LINE_8_CONST;
extern const int COLOR_BGR2RGB_CONST;
extern const int CV_32F_CONST;

// Image I/O operations
cv::Mat* load_image(const std::string& filename, int flags = 1);
bool save_image(const std::string& filename, cv::Mat* img);

// Window management
void create_window(const std::string& winname, int flags = WINDOW_AUTOSIZE_CONST);
void show_image(const std::string& winname, cv::Mat* img);
int wait_key(int delay = 0);
void destroy_window(const std::string& winname);
void destroy_all_windows();

// Drawing operations
void draw_text(cv::Mat* img, const std::string& text, cv::Point org, 
              int fontFace, double fontScale, cv::Scalar color, 
              int thickness = 1, int lineType = LINE_8_CONST, bool bottomLeftOrigin = false);
void draw_rectangle(cv::Mat* img, cv::Rect rect, cv::Scalar color, 
                   int thickness = 1, int lineType = LINE_8_CONST, int shift = 0);

// Image processing
void resize_image(const cv::Mat* src, cv::Mat* dst, cv::Size size, 
                 double fx = 0, double fy = 0, int interpolation = 1);
void cvt_color(const cv::Mat* src, cv::Mat* dst, int code);
void subtract_scalar(cv::Mat* img, const cv::Scalar& val);
void divide_scalar(cv::Mat* img, const cv::Scalar& val);