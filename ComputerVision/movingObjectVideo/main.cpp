#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <mutex>
#include <string>
#include <iomanip> // For std::fixed and std::setprecision

class MotionDetector {
private:
    // Parameters for ORB detector
    int nFeatures;
    float matchingThreshold;
    int minInliers;
    
    // Mutex for thread safety when accessing class members
    std::mutex detectorMutex;

    /**
     * Calculates velocity vectors for matched keypoints
     * @param points1 Points in the first image
     * @param points2 Corresponding points in the second image
     * @param timeInterval Time interval between images in seconds
     * @return Vector of velocity magnitudes and directions
     * 
     * Time Complexity: O(n) where n is the number of points
     * Memory Complexity: O(n) for storing velocity vectors
     */
    std::vector<std::pair<float, cv::Point2f>> calculateVelocities(
        const std::vector<cv::Point2f>& points1, 
        const std::vector<cv::Point2f>& points2,
        float timeInterval) {
        
        // Validate input
        if (timeInterval <= 0) {
            throw std::invalid_argument("Time interval must be positive");
        }
        
        if (points1.size() != points2.size()) {
            throw std::invalid_argument("Point arrays must have the same size");
        }
        
        std::vector<std::pair<float, cv::Point2f>> velocities;
        velocities.reserve(points1.size());
        
        for (size_t i = 0; i < points1.size(); i++) {
            // Calculate displacement vector
            cv::Point2f displacement = points2[i] - points1[i];
            
            // Calculate velocity (displacement / time)
            cv::Point2f velocity = displacement * (1.0f / timeInterval);
            
            // Calculate velocity magnitude (in pixels per second)
            float magnitude = cv::norm(velocity);
            
            // Store velocity magnitude and velocity vector
            velocities.push_back(std::make_pair(magnitude, velocity));
        }
        
        return velocities;
    }

public:
    /**
     * Constructor with default parameters
     * @param nFeatures Number of features to extract (default: 500)
     * @param matchingThreshold Distance threshold for good matches (default: 0.75)
     * @param minInliers Minimum number of inliers to consider motion valid (default: 10)
     */
    MotionDetector(int nFeatures = 500, float matchingThreshold = 0.75, int minInliers = 10) 
        : nFeatures(nFeatures), matchingThreshold(matchingThreshold), minInliers(minInliers) {}
    
    /**
     * Detects motion between two images, calculates velocities, and returns a visualization
     * @param img1 First image (previous frame)
     * @param img2 Second image (current frame)
     * @param motionMask Output binary mask showing motion areas
     * @param timeInterval Time interval between images in seconds (default: 0.5)
     * @return Visualization of the motion detection with velocity information
     * 
     * Time Complexity: O(n log n) where n is the number of keypoints
     * Memory Complexity: O(n) for storing keypoints, descriptors, and velocity data
     */
    cv::Mat detectMotion(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& motionMask, float timeInterval = 0.5f) {
        // Input validation
        if (img1.empty() || img2.empty()) {
            throw std::invalid_argument("Input images cannot be empty");
        }
        
        if (img1.size() != img2.size()) {
            throw std::invalid_argument("Input images must have the same dimensions");
        }
        
        if (timeInterval <= 0) {
            throw std::invalid_argument("Time interval must be positive");
        }
        
        // Lock mutex to ensure thread safety when using class members
        std::lock_guard<std::mutex> lock(detectorMutex);
        // This lock is needed because multiple threads could call detectMotion simultaneously
        // and modify the same ORB detector parameters
        
        // Convert images to grayscale if they are not already
        cv::Mat gray1, gray2;
        if (img1.channels() == 3) {
            cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
        } else {
            gray1 = img1.clone();
            gray2 = img2.clone();
        }
        
        // Initialize ORB detector
        cv::Ptr<cv::ORB> orb = cv::ORB::create(nFeatures);
        
        // Detect keypoints and compute descriptors
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        
        orb->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);
        
        // Handle edge case: no keypoints detected
        if (keypoints1.empty() || keypoints2.empty() || 
            descriptors1.empty() || descriptors2.empty()) {
            std::cout << "Warning: No keypoints detected in one or both images" << std::endl;
            motionMask = cv::Mat::zeros(img1.size(), CV_8UC1);
            return img2.clone();
        }
        
        // Match descriptors using Brute Force Hamming distance
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
        
        // Filter matches using Lowe's ratio test
        std::vector<cv::DMatch> goodMatches;
        for (const auto& match : knnMatches) {
            if (match.size() < 2) continue;
            
            if (match[0].distance < matchingThreshold * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
        }
        
        // Create point correspondences for finding homography
        std::vector<cv::Point2f> points1, points2;
        for (const auto& match : goodMatches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
        
        // Calculate velocities for matched keypoints
        std::vector<std::pair<float, cv::Point2f>> velocities = calculateVelocities(points1, points2, timeInterval);
        
        // Initialize motion mask
        motionMask = cv::Mat::zeros(img1.size(), CV_8UC1);
        
        // Create a copy of the second image for visualization
        cv::Mat result = img2.clone();
        
        // Draw title with time interval information
        std::stringstream titleStream;
        titleStream << std::fixed << std::setprecision(3);
        titleStream << "Motion detection - Time interval: " << timeInterval << "s";
        std::string title = titleStream.str();
        cv::putText(result, title, cv::Point(20, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
        
        // If we have enough good matches, compute homography and find moving points
        if (goodMatches.size() >= 4) {
            // Find homography matrix (perspective transformation between images)
            std::vector<uchar> inliersMask;
            cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, 3.0, inliersMask);
            
            if (!H.empty()) {
                // Count inliers (points that conform to the homography)
                int inliersCount = cv::countNonZero(inliersMask);
                
                if (inliersCount >= minInliers) {
                    // Warp first image to align with second image
                    cv::Mat warpedImg1;
                    cv::warpPerspective(img1, warpedImg1, H, img1.size());
                    
                    // Calculate absolute difference between warped img1 and img2
                    // This highlights the actual moving objects
                    cv::Mat diff;
                    cv::absdiff(warpedImg1, img2, diff);
                    
                    // Convert to grayscale if needed
                    cv::Mat diffGray;
                    if (diff.channels() == 3) {
                        cv::cvtColor(diff, diffGray, cv::COLOR_BGR2GRAY);
                    } else {
                        diffGray = diff.clone();
                    }
                    
                    // Threshold to get binary mask of moving regions
                    cv::threshold(diffGray, motionMask, 30, 255, cv::THRESH_BINARY);
                    
                    // Apply morphological operations to clean up the mask
                    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
                    cv::morphologyEx(motionMask, motionMask, cv::MORPH_OPEN, kernel);
                    cv::morphologyEx(motionMask, motionMask, cv::MORPH_CLOSE, kernel);
                    
                    // Find contours of moving objects
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(motionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                    
                    // Draw contours on the result image
                    cv::drawContours(result, contours, -1, cv::Scalar(0, 0, 255), 2);
                    
                    // Draw bounding boxes around moving objects with velocity information
                    for (const auto& contour : contours) {
                        // Filter small contours
                        if (cv::contourArea(contour) < 100) continue;
                        
                        cv::Rect boundingBox = cv::boundingRect(contour);
                        cv::rectangle(result, boundingBox, cv::Scalar(0, 255, 0), 2);
                        
                        // Calculate average velocity for points in this bounding box
                        float totalVelocity = 0.0f;
                        int pointsInBox = 0;
                        cv::Point2f avgDirection(0, 0);
                        
                        for (size_t i = 0; i < goodMatches.size(); i++) {
                            const cv::Point2f& pt = points2[i];
                            if (boundingBox.contains(pt)) {
                                totalVelocity += velocities[i].first;
                                avgDirection += velocities[i].second;
                                pointsInBox++;
                            }
                        }
                        
                        // Add velocity information
                        std::string label = "Moving Object";
                        if (pointsInBox > 0) {
                            float avgVelocity = totalVelocity / pointsInBox;
                            avgDirection = avgDirection * (1.0f / pointsInBox);
                            
                            std::stringstream ss;
                            ss << std::fixed << std::setprecision(1);
                            ss << "Velocity: " << avgVelocity << " px/s";
                            label = ss.str();
                            
                            // Draw a prominent velocity vector for this object
                            cv::Point center = (boundingBox.tl() + boundingBox.br()) * 0.5;
                            float arrowLength = std::min(80.0f, avgVelocity / 2);  // Increased max length
                            cv::Point2f arrowDirection = avgDirection;
                            if (cv::norm(arrowDirection) > 0) {
                                arrowDirection = arrowDirection * (arrowLength / cv::norm(arrowDirection));
                                // Convert center to Point2f for consistent type with arrowDirection
                                cv::Point2f centerf(center.x, center.y);
                                cv::arrowedLine(result, centerf, centerf + arrowDirection, 
                                              cv::Scalar(0, 255, 0), 3, cv::LINE_AA, 0, 0.4);  // Thicker, more visible arrow
                                
                                // Add velocity magnitude at the arrow tip
                                std::stringstream ssTip;
                                ssTip << std::fixed << std::setprecision(1) << avgVelocity;
                                cv::putText(result, ssTip.str() + " px/s", 
                                           centerf + arrowDirection + cv::Point2f(5, 0),
                                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                            }
                        }
                        
                        cv::putText(result, label, 
                                   cv::Point(boundingBox.x, boundingBox.y - 10),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
        }
        
        // Display frame and time information
        std::stringstream statsStream;
        statsStream << std::fixed << std::setprecision(3);
        statsStream << "Time interval: " << timeInterval << " seconds";
        std::string statsText = statsStream.str();
        
        cv::putText(result, statsText, cv::Point(20, result.rows - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        
        return result;
    }
    
    /**
     * Updates the detector parameters
     * @param newFeatures New number of features
     * @param newThreshold New matching threshold
     * @param newMinInliers New minimum inliers threshold
     */
    void updateParameters(int newFeatures, float newThreshold, int newMinInliers) {
        std::lock_guard<std::mutex> lock(detectorMutex);
        // This lock is needed because parameters could be updated while
        // detectMotion is being called in another thread
        
        nFeatures = newFeatures;
        matchingThreshold = newThreshold;
        minInliers = newMinInliers;
    }
};

/**
 * Process a video file and track object velocities
 * Time Complexity: O(n log n * f) where n is the number of keypoints and f is the number of frames
 * Memory Complexity: O(n) for storing keypoints, descriptors, and velocity data
 */
int main(int argc, char** argv) {
    try {
        // Check command line arguments
        if (argc != 2 && argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <video_file> [frame_skip]" << std::endl;
            return -1;
        }
        
        // Get video file path
        std::string videoPath = argv[1];
        
        // Parse optional frame skip parameter
        int frameSkip = 1; // Process every frame by default
        if (argc == 3) {
            try {
                frameSkip = std::stoi(argv[2]);
                if (frameSkip < 1) {
                    throw std::invalid_argument("Frame skip must be a positive integer");
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing frame skip value: " << e.what() << std::endl;
                std::cerr << "Using default frame skip of 1" << std::endl;
                frameSkip = 1;
            }
        }
        
        // Open the video file
        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
            return -1;
        }
        
        // Get video properties
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        
        // Calculate time interval between frames based on FPS and frame skip
        float timeInterval = frameSkip / static_cast<float>(fps);
        
        std::cout << "Video info:" << std::endl;
        std::cout << "- Resolution: " << width << "x" << height << std::endl;
        std::cout << "- FPS: " << fps << std::endl;
        std::cout << "- Frame count: " << frameCount << std::endl;
        std::cout << "- Time interval between processed frames: " << timeInterval << " seconds" << std::endl;
        
        // Create output video writer
        cv::VideoWriter writer("motion_velocity_output.mp4", 
                              cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                              fps / frameSkip, // Output at adjusted FPS
                              cv::Size(width, height));
        
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not create output video file." << std::endl;
            return -1;
        }
        
        // Create motion detector
        MotionDetector detector(1000, 0.7, 15);
        
        // Read first frame
        cv::Mat prevFrame, currFrame;
        if (!cap.read(prevFrame)) {
            std::cerr << "Error: Could not read the first frame." << std::endl;
            return -1;
        }
        
        // Skip frames if needed to get to the next frame to process
        for (int i = 0; i < frameSkip - 1; i++) {
            if (!cap.read(currFrame)) {
                std::cerr << "Error: Could not read frame after skipping." << std::endl;
                return -1;
            }
        }
        
        // Create windows for display
        cv::namedWindow("Original Video", cv::WINDOW_NORMAL);
        cv::namedWindow("Motion Detection with Velocity", cv::WINDOW_NORMAL);
        
        // Process the video frame by frame
        int frameNumber = 0;
        bool paused = false;
        
        while (true) {
            // Read the next frame
            if (!paused) {
                if (!cap.read(currFrame)) {
                    break; // End of video
                }
                
                // Skip frames if needed
                for (int i = 0; i < frameSkip - 1; i++) {
                    if (!cap.read(currFrame)) {
                        break;
                    }
                }
                
                // Process frames
                cv::Mat motionMask;
                cv::Mat result;
                
                try {
                    // Add frame number info
                    frameNumber++;
                    
                    // Process the frame pair
                    result = detector.detectMotion(prevFrame, currFrame, motionMask, timeInterval);
                    
                    // Add frame number to result
                    std::stringstream frameInfo;
                    frameInfo << "Frame: " << frameNumber;
                    cv::putText(result, frameInfo.str(), 
                               cv::Point(result.cols - 150, 30), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
                    
                    // Display the original and processed frames
                    cv::imshow("Original Video", currFrame);
                    cv::imshow("Motion Detection with Velocity", result);
                    
                    // Write the frame to output video
                    writer.write(result);
                    
                    // Current frame becomes previous frame for next iteration
                    prevFrame = currFrame.clone();
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing frame " << frameNumber << ": " << e.what() << std::endl;
                    // If an error occurs, still continue with the next frame
                    prevFrame = currFrame.clone();
                }
            }
            
            // Wait for keyboard input (with a short delay for smooth playback)
            int key = cv::waitKey(30);
            
            // Handle key presses
            if (key == 27) { // ESC key
                break; // Exit the loop
            }
            else if (key == ' ') { // Space key for pause/resume
                paused = !paused;
                std::cout << (paused ? "Paused" : "Resumed") << std::endl;
            }
            else if (key == 's' && paused) { // 's' key to step forward when paused
                if (!cap.read(currFrame)) {
                    break; // End of video
                }
                
                // Skip frames if needed
                for (int i = 0; i < frameSkip - 1; i++) {
                    if (!cap.read(currFrame)) {
                        break;
                    }
                }
                
                // Process frame
                cv::Mat motionMask;
                cv::Mat result = detector.detectMotion(prevFrame, currFrame, motionMask, timeInterval);
                
                // Add frame number
                frameNumber++;
                std::stringstream frameInfo;
                frameInfo << "Frame: " << frameNumber;
                cv::putText(result, frameInfo.str(), 
                           cv::Point(result.cols - 150, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
                
                // Display and write
                cv::imshow("Original Video", currFrame);
                cv::imshow("Motion Detection with Velocity", result);
                writer.write(result);
                
                // Update previous frame
                prevFrame = currFrame.clone();
            }
        }
        
        // Release resources
        cap.release();
        writer.release();
        cv::destroyAllWindows();
        
        std::cout << "Motion detection with velocity tracking completed successfully." << std::endl;
        std::cout << "Output saved as 'motion_velocity_output.mp4'" << std::endl;
        std::cout << "Processed " << frameNumber << " frames." << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}