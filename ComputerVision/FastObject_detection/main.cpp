#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

class FMODetector {
private:
    // Parameters
    int threshold_diff;           // Threshold for background difference
    int min_area;                 // Minimum area to consider an object
    float max_aspect_ratio;       // Maximum aspect ratio for FMO candidates
    int history_size;             // Background model history size
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor;
    cv::Mat background_model;
    std::vector<cv::Mat> frame_history;

public:
    FMODetector(int threshold_diff = 30, 
                int min_area = 50, 
                float max_aspect_ratio = 5.0,
                int history_size = 20) {
        this->threshold_diff = threshold_diff;
        this->min_area = min_area;
        this->max_aspect_ratio = max_aspect_ratio;
        this->history_size = history_size;
        
        // Initialize background subtractor
        bg_subtractor = cv::createBackgroundSubtractorMOG2(history_size, 16, false);
    }

    struct FMOObject {
        cv::Rect bbox;            // Bounding box
        cv::Point2f direction;    // Movement direction vector
        float speed;              // Estimated speed
        cv::Mat appearance;       // Visual appearance of the object
    };

    std::vector<FMOObject> detect(const cv::Mat& frame) {
        std::vector<FMOObject> detected_fmos;
        
        if (frame.empty()) {
            std::cerr << "Empty frame provided to FMO detector" << std::endl;
            return detected_fmos;
        }
        
        // Convert to grayscale for processing
        cv::Mat gray_frame;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        } else {
            gray_frame = frame.clone();
        }
        
        // Update frame history
        updateFrameHistory(gray_frame);
        
        // Step 1: Background subtraction
        cv::Mat foreground_mask;
        bg_subtractor->apply(gray_frame, foreground_mask);
        
        // Step 2: Apply morphological operations to clean up the mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(foreground_mask, foreground_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(foreground_mask, foreground_mask, cv::MORPH_CLOSE, kernel);
        
        // Step 3: Find contours in the foreground mask
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(foreground_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Step 4: Process each contour
        for (const auto& contour : contours) {
            // Analyze contour shape
            double area = cv::contourArea(contour);
            
            // Filter by area
            if (area < min_area) continue;
            
            // Get bounding rectangle
            cv::Rect bbox = cv::boundingRect(contour);
            
            // Filter by aspect ratio
            float aspect_ratio = static_cast<float>(bbox.width) / bbox.height;
            if (aspect_ratio > max_aspect_ratio || aspect_ratio < 1.0/max_aspect_ratio) {
                // Elongated objects are often FMOs due to motion blur
                
                // Compute motion features using temporal information
                cv::Point2f direction;
                float speed;
                computeMotionFeatures(gray_frame, bbox, direction, speed);
                
                // If speed is high enough, consider it an FMO
                if (speed > 5.0) {  // Adjustable threshold
                    FMOObject fmo;
                    fmo.bbox = bbox;
                    fmo.direction = direction;
                    fmo.speed = speed;
                    fmo.appearance = frame(bbox).clone();
                    
                    detected_fmos.push_back(fmo);
                }
            }
        }
        
        // Step 5: Track existing FMOs across frames (could be extended)
        
        return detected_fmos;
    }

private:
    void updateFrameHistory(const cv::Mat& frame) {
        // Add current frame to history
        frame_history.push_back(frame.clone());
        
        // Keep history size limited
        if (frame_history.size() > history_size) {
            frame_history.erase(frame_history.begin());
        }
    }
    
    void computeMotionFeatures(const cv::Mat& current_frame, const cv::Rect& bbox, 
                              cv::Point2f& direction, float& speed) {
        // Initialize with default values
        direction = cv::Point2f(0, 0);
        speed = 0.0f;
        
        // Need at least 2 frames for motion estimation
        if (frame_history.size() < 2) return;
        
        // Expand search area
        cv::Rect search_area = expandRect(bbox, current_frame.size(), 1.5);
        
        // Compare with previous frame
        cv::Mat prev_frame = frame_history[frame_history.size() - 2];
        
        // Calculate optical flow in the region of interest
        std::vector<cv::Point2f> prev_points, curr_points;
        std::vector<uchar> status;
        std::vector<float> err;
        
        // Extract good features to track in the current bounding box
        cv::Mat roi = current_frame(search_area);
        cv::goodFeaturesToTrack(roi, curr_points, 100, 0.01, 10);
        
        // Adjust points to full frame coordinates
        for (auto& pt : curr_points) {
            pt.x += search_area.x;
            pt.y += search_area.y;
        }
        
        if (curr_points.empty()) return;
        
        // Calculate optical flow
        cv::calcOpticalFlowPyrLK(prev_frame, current_frame, curr_points, prev_points, status, err);
        
        // Calculate average motion vector
        cv::Point2f avg_motion(0, 0);
        int valid_points = 0;
        
        for (size_t i = 0; i < curr_points.size(); i++) {
            if (status[i]) {
                avg_motion += curr_points[i] - prev_points[i];
                valid_points++;
            }
        }
        
        if (valid_points > 0) {
            avg_motion /= valid_points;
            direction = avg_motion;
            speed = cv::norm(avg_motion);
        }
    }
    
    cv::Rect expandRect(const cv::Rect& rect, const cv::Size& img_size, float scale) {
        cv::Rect expanded;
        
        int width_diff = static_cast<int>(rect.width * (scale - 1));
        int height_diff = static_cast<int>(rect.height * (scale - 1));
        
        expanded.x = std::max(0, rect.x - width_diff/2);
        expanded.y = std::max(0, rect.y - height_diff/2);
        expanded.width = std::min(img_size.width - expanded.x, rect.width + width_diff);
        expanded.height = std::min(img_size.height - expanded.y, rect.height + height_diff);
        
        return expanded;
    }
};

// Example usage
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./fmo_detector <video_path>" << std::endl;
        return -1;
    }
    
    // Open video \\wsl.localhost\Ubuntu\home\jiwokim\Deeplearning\yolov5
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }
    
    // Create FMO detector
    FMODetector detector(25, 100, 4.0, 15);
    
    cv::Mat frame;
    while (cap.read(frame)) {
        // Detect FMOs
        std::vector<FMODetector::FMOObject> fmos = detector.detect(frame);
        
        // Visualize results
        for (const auto& fmo : fmos) {
            // Draw bounding box
            cv::rectangle(frame, fmo.bbox, cv::Scalar(0, 255, 0), 2);
            
            // Draw motion vector
            cv::Point center(fmo.bbox.x + fmo.bbox.width/2, fmo.bbox.y + fmo.bbox.height/2);
            cv::line(frame, center, 
                    cv::Point(center.x + fmo.direction.x*5, center.y + fmo.direction.y*5),
                    cv::Scalar(255, 0, 0), 2);
            
            // Display speed
            std::string speed_text = "Speed: " + std::to_string(fmo.speed);
            cv::putText(frame, speed_text, cv::Point(fmo.bbox.x, fmo.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                std::cout << speed_text <<std::endl;
        }
        
        // Display result
        cv::imshow("FMO Detection", frame);
        
        // Exit on ESC key
        if (cv::waitKey(30) == 27) break;
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}