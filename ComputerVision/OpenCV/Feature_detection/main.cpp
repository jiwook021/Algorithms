#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main() {
    // Load two images in color
    cv::Mat Img1Color = cv::imread("object.jpg", cv::IMREAD_COLOR);
    cv::Mat Img2Color = cv::imread("scene.jpg", cv::IMREAD_COLOR);
    
    // Check if images loaded successfully
    if (Img1Color.empty() || Img2Color.empty()) {
        std::cout << "Could not open or find the images" << std::endl;
        return -1;
    }
    
    // Create grayscale versions for feature detection
    cv::Mat Img1Gray, Img2Gray;
    cv::cvtColor(Img1Color, Img1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(Img2Color, Img2Gray, cv::COLOR_BGR2GRAY);
    
    // Initialize the feature detector (ORB)
    cv::Ptr<cv::ORB> Detector = cv::ORB::create();
    
    // Detect keypoints in grayscale images
    std::vector<cv::KeyPoint> Keypoints1, Keypoints2;
    Detector->detect(Img1Gray, Keypoints1);
    Detector->detect(Img2Gray, Keypoints2);
    
    // Calculate descriptors
    cv::Mat Descriptors1, Descriptors2;
    Detector->compute(Img1Gray, Keypoints1, Descriptors1);
    Detector->compute(Img2Gray, Keypoints2, Descriptors2);
    
    // Match descriptors using Brute Force matcher
    cv::BFMatcher Matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> Matches;
    Matcher.match(Descriptors1, Descriptors2, Matches);
    
    // Sort matches by distance (ascending order)
    std::sort(Matches.begin(), Matches.end(), 
              [](const cv::DMatch& a, const cv::DMatch& b) {
                  return a.distance < b.distance;
              });
    
    // Keep only the top 30 good matches
    const int NumGoodMatches = std::min(30, static_cast<int>(Matches.size()));
    Matches.erase(Matches.begin() + NumGoodMatches, Matches.end());
    
    // Draw the matches using color images
    cv::Mat ImgMatches;
    cv::drawMatches(Img1Color, Keypoints1, Img2Color, Keypoints2, Matches, ImgMatches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    // Draw detected keypoints in color images
    cv::Mat ImgKeypoints1, ImgKeypoints2;
    cv::drawKeypoints(Img1Color, Keypoints1, ImgKeypoints1, cv::Scalar::all(-1), 
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(Img2Color, Keypoints2, ImgKeypoints2, cv::Scalar::all(-1), 
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    // Resize images for display
    double Scale = 0.5; // Resize to 50% of original size
    cv::Mat imgKeypoints1_resized, imgKeypoints2_resized, imgMatches_resized;
    cv::resize(ImgKeypoints1, imgKeypoints1_resized, cv::Size(), Scale, Scale);
    cv::resize(ImgKeypoints2, imgKeypoints2_resized, cv::Size(), Scale, Scale);
    cv::resize(ImgMatches, imgMatches_resized, cv::Size(), Scale, Scale);
    
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