#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // Load two images in color
    cv::Mat img1_color = cv::imread("object.jpg", cv::IMREAD_COLOR);
    cv::Mat img2_color = cv::imread("scene.jpg", cv::IMREAD_COLOR);
    
    // Check if images loaded successfully
    if (img1_color.empty() || img2_color.empty()) {
        std::cout << "Could not open or find the images" << std::endl;
        return -1;
    }
    
    // Create grayscale versions for feature detection
    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1_color, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2_color, img2_gray, cv::COLOR_BGR2GRAY);
    
    // Initialize the feature detector (ORB)
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    
    // Detect keypoints in grayscale images
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    detector->detect(img1_gray, keypoints1);
    detector->detect(img2_gray, keypoints2);
    
    // Calculate descriptors
    cv::Mat descriptors1, descriptors2;
    detector->compute(img1_gray, keypoints1, descriptors1);
    detector->compute(img2_gray, keypoints2, descriptors2);
    
    // Match descriptors using Brute Force matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    // Sort matches by distance (ascending order)
    std::sort(matches.begin(), matches.end(), 
              [](const cv::DMatch& a, const cv::DMatch& b) {
                  return a.distance < b.distance;
              });
    
    // Keep only the top 30 good matches
    const int numGoodMatches = std::min(30, static_cast<int>(matches.size()));
    matches.erase(matches.begin() + numGoodMatches, matches.end());
    
    // Draw the matches using color images
    cv::Mat imgMatches;
    cv::drawMatches(img1_color, keypoints1, img2_color, keypoints2, matches, imgMatches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // Draw detected keypoints in color images
    cv::Mat imgKeypoints1, imgKeypoints2;
    cv::drawKeypoints(img1_color, keypoints1, imgKeypoints1, cv::Scalar::all(-1), 
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(img2_color, keypoints2, imgKeypoints2, cv::Scalar::all(-1), 
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    // Resize images for display
    double scale = 0.5; // Resize to 50% of original size
    cv::Mat imgKeypoints1_resized, imgKeypoints2_resized, imgMatches_resized;
    cv::resize(imgKeypoints1, imgKeypoints1_resized, cv::Size(), scale, scale);
    cv::resize(imgKeypoints2, imgKeypoints2_resized, cv::Size(), scale, scale);
    cv::resize(imgMatches, imgMatches_resized, cv::Size(), scale, scale);
    
    // Create resizable windows
    cv::namedWindow("Keypoints 1", cv::WINDOW_NORMAL);
    cv::namedWindow("Keypoints 2", cv::WINDOW_NORMAL);
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    
    // Display the resized images
    cv::imshow("Keypoints 1", imgKeypoints1_resized);
    cv::imshow("Keypoints 2", imgKeypoints2_resized);
    cv::imshow("Matches", imgMatches_resized);
    
    cv::waitKey(0);
    return 0;
}