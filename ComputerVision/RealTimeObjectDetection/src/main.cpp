/**
 * @file main.cpp
 * @brief Driver for RealTimeObjectDetection
 */

#include "RealTimeObjectDetection.hpp"

 int main(int argc, char* argv[]) {
     try {
         // Default parameters
         std::string VideoPath;
         std::string ModelPath = "models/yolov8s.onnx";
         float ConfThreshold = 0.25f;
         float NmsThreshold = 0.45f;
         int CameraId = 0;
         bool ShowDetections = true;
         bool ShowTracks = true;
         bool ShowFPS = true;
         std::string SavePath;
         cv::Rect RoiZone;
         
         // Parse command line arguments
         if (!ParseArgs(argc, argv, VideoPath, ModelPath, ConfThreshold, NmsThreshold,
                       CameraId, ShowDetections, ShowTracks, ShowFPS, SavePath, RoiZone)) {
             return 1;
         }
         
         // Initialize detector
         std::cout << "Initializing detector with model: " << ModelPath << std::endl;
         auto detector = std::make_shared<::Detector>(ModelPath, ConfThreshold, NmsThreshold);
         
         // Initialize tracker
         std::cout << "Initializing tracker" << std::endl;
         auto Tracker = std::make_shared<KalmanTracker>();
         
         // Open video source
         cv::VideoCapture VideoSource;
         if (VideoPath.empty()) {
             std::cout << "Opening camera device: " << CameraId << std::endl;
             VideoSource.open(CameraId);
         } else {
             std::cout << "Opening video file: " << VideoPath << std::endl;
             VideoSource.open(VideoPath);
         }
         
         if (!VideoSource.isOpened()) {
             std::cerr << "Error: Could not open video source" << std::endl;
             return 1;
         }
         
         // Create frame queue for Producer-Consumer pattern
         ThreadSafeQueue<std::shared_ptr<FrameData>> FrameQueue;
         
         // Create atomic flag for thread synchronization
         std::atomic<bool> Running(true);
         
         // Create and start producer thread
         std::thread Producer(ProducerThread, std::ref(VideoSource), detector, std::ref(FrameQueue), std::ref(Running));
         
         // Create and start consumer thread
         std::thread Consumer(ConsumerThread, detector, Tracker, std::ref(FrameQueue), std::ref(Running), SavePath, RoiZone);
         
         // Wait for threads to finish
         Producer.join();
         Consumer.join();
         
         // Close video source
         VideoSource.release();
         
         // Run tests if requested
         bool RunTestScenarios = false;
         for (int i = 1; i < argc; i++) {
             if (std::string(argv[i]) == "--run-tests") {
                 RunTestScenarios = true;
                 break;
             }
         }
         
         if (RunTestScenarios) {
             RunTests(detector, Tracker);
         }
         
         std::cout << "Program completed successfully" << std::endl;
         return 0;
         
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
 }