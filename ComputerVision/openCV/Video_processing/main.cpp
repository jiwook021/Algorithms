#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open a video file
    cv::VideoCapture cap("video.mp4");
    
    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cout << "Error opening video file" << std::endl;
        return -1;
    }
    
    // Get video properties
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Video properties: " << frameWidth << "x" << frameHeight 
              << ", " << fps << " FPS" << std::endl;
    
    // Create VideoWriter object
    cv::VideoWriter writer("output_video.avi", 
                          cv::VideoWriter::fourcc('M','J','P','G'), 
                          fps, 
                          cv::Size(frameWidth, frameHeight));
    
    // Process video frame by frame
    while (true) {
        cv::Mat frame;
        
        // Capture frame-by-frame
        cap >> frame;
        
        // Break the loop if no more frames
        if (frame.empty())
            break;
        
        // Example processing: Convert to grayscale
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        cv::cvtColor(grayFrame, grayFrame, cv::COLOR_GRAY2BGR); // Convert back for writing
        
        // Write the processed frame
        writer.write(grayFrame);
        
        // Display the frames
        cv::imshow("Original Frame", frame);
        cv::imshow("Processed Frame", grayFrame);
        
        // Press ESC to exit
        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }
    
    // Release the resources
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    
    return 0;
}