/**
 * @file main.cpp
 * @brief Main entry point for the object detection and tracking application
 */

 #include <iostream>
 #include <chrono>
 #include <thread>
 #include <atomic>
 #include <memory>
 #include <filesystem>
 #include <stdexcept>
 #include <opencv2/opencv.hpp>
 
 #include "Detector.h"
 #include "KalmanTracker.h"
 #include "include/ThreadSafeQueue.h"
 #include "Viewer.h"
 
 namespace fs = std::filesystem;
 
 /**
  * @struct FrameData
  * @brief Structure to hold frame data for processing
  */
 struct FrameData {
     cv::Mat frame;               ///< Original frame
     std::vector<Detection> detections; ///< Detected objects
     int frameIndex;              ///< Frame index
     float inferenceTime;         ///< Inference time in milliseconds
     
     FrameData(cv::Mat f, int idx) 
         : frame(std::move(f)), frameIndex(idx), inferenceTime(0.0f) {}
 };
 
 /**
  * @brief Display command line usage information
  * 
  * @param programName Name of the program
  */
 void printUsage(const std::string& programName) {
     std::cout << "Usage: " << programName << " [options]\n"
               << "Options:\n"
               << "  --video <path>       : Path to video file (default: use camera)\n"
               << "  --model <path>       : Path to YOLOv8 ONNX model (default: models/yolov8s.onnx)\n"
               << "  --conf <threshold>   : Confidence threshold (default: 0.25)\n"
               << "  --nms <threshold>    : NMS threshold (default: 0.45)\n"
               << "  --camera <id>        : Camera device ID (default: 0)\n"
               << "  --show-detections    : Show raw detections (default: true)\n"
               << "  --show-tracks        : Show tracked objects (default: true)\n"
               << "  --show-fps           : Show FPS and performance metrics (default: true)\n"
               << "  --help               : Show this help message\n";
 }
 
 /**
  * @brief Parse command line arguments
  * 
  * @param argc Argument count
  * @param argv Argument values
  * @param videoPath Path to video file
  * @param modelPath Path to YOLOv8 ONNX model
  * @param confThreshold Confidence threshold
  * @param nmsThreshold NMS threshold
  * @param cameraId Camera device ID
  * @param showDetections Flag to show raw detections
  * @param showTracks Flag to show tracked objects
  * @param showFPS Flag to show FPS and performance metrics
  * @return true if arguments are valid, false otherwise
  */
 bool parseArgs(int argc, char* argv[],
                std::string& videoPath,
                std::string& modelPath,
                float& confThreshold,
                float& nmsThreshold,
                int& cameraId,
                bool& showDetections,
                bool& showTracks,
                bool& showFPS) {
     for (int i = 1; i < argc; i++) {
         std::string arg = argv[i];
         
         if (arg == "--help") {
             printUsage(argv[0]);
             return false;
         } else if (arg == "--video" && i + 1 < argc) {
             videoPath = argv[++i];
         } else if (arg == "--model" && i + 1 < argc) {
             modelPath = argv[++i];
         } else if (arg == "--conf" && i + 1 < argc) {
             try {
                 confThreshold = std::stof(argv[++i]);
                 if (confThreshold < 0.0f || confThreshold > 1.0f) {
                     std::cerr << "Error: Confidence threshold must be between 0 and 1" << std::endl;
                     return false;
                 }
             } catch (const std::exception& e) {
                 std::cerr << "Error parsing confidence threshold: " << e.what() << std::endl;
                 return false;
             }
         } else if (arg == "--nms" && i + 1 < argc) {
             try {
                 nmsThreshold = std::stof(argv[++i]);
                 if (nmsThreshold < 0.0f || nmsThreshold > 1.0f) {
                     std::cerr << "Error: NMS threshold must be between 0 and 1" << std::endl;
                     return false;
                 }
             } catch (const std::exception& e) {
                 std::cerr << "Error parsing NMS threshold: " << e.what() << std::endl;
                 return false;
             }
         } else if (arg == "--camera" && i + 1 < argc) {
             try {
                 cameraId = std::stoi(argv[++i]);
                 if (cameraId < 0) {
                     std::cerr << "Error: Camera ID must be non-negative" << std::endl;
                     return false;
                 }
             } catch (const std::exception& e) {
                 std::cerr << "Error parsing camera ID: " << e.what() << std::endl;
                 return false;
             }
         } else if (arg == "--show-detections") {
             showDetections = true;
         } else if (arg == "--no-show-detections") {
             showDetections = false;
         } else if (arg == "--show-tracks") {
             showTracks = true;
         } else if (arg == "--no-show-tracks") {
             showTracks = false;
         } else if (arg == "--show-fps") {
             showFPS = true;
         } else if (arg == "--no-show-fps") {
             showFPS = false;
         } else {
             std::cerr << "Unknown argument: " << arg << std::endl;
             printUsage(argv[0]);
             return false;
         }
     }
     
     // Validate model path
     if (!fs::exists(modelPath)) {
         std::cerr << "Error: Model file not found: " << modelPath << std::endl;
         return false;
     }
     
     // Validate video path if provided
     if (!videoPath.empty() && !fs::exists(videoPath)) {
         std::cerr << "Error: Video file not found: " << videoPath << std::endl;
         return false;
     }
     
     return true;
 }
 
 /**
  * @brief Producer thread function
  * 
  * This function reads frames from the video source, performs object detection,
  * and pushes the results to the queue for the consumer thread.
  * 
  * @param videoSource Video source (file or camera)
  * @param detector Object detector
  * @param frameQueue Queue for passing frames to the consumer
  * @param running Atomic flag indicating if the application is running
  */
 void producerThread(cv::VideoCapture& videoSource,
                    std::shared_ptr<Detector> detector,
                    ThreadSafeQueue<std::shared_ptr<FrameData>>& frameQueue,
                    std::atomic<bool>& running) {
     int frameIndex = 0;
     cv::Mat frame;
     
     // Set CUDA stream for current thread
     cv::cuda::Stream stream;
     
     while (running) {
         // Read frame
         videoSource >> frame;
         
         // Check if frame is valid
         if (frame.empty()) {
             std::cout << "End of video or failed to read frame" << std::endl;
             break;
         }
         
         // Create frame data
         auto frameData = std::make_shared<FrameData>(frame.clone(), frameIndex++);
         
         try {
             // Detect objects
             auto startTime = std::chrono::high_resolution_clock::now();
             frameData->detections = detector->detect(frameData->frame);
             auto endTime = std::chrono::high_resolution_clock::now();
             
             // Calculate inference time
             frameData->inferenceTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
             
             // Push frame data to queue
             frameQueue.push(frameData);
             
         } catch (const std::exception& e) {
             std::cerr << "Error in producer thread: " << e.what() << std::endl;
         }
     }
     
     // Signal end of frames
     frameQueue.done();
     std::cout << "Producer thread finished" << std::endl;
 }
 
 /**
  * @brief Consumer thread function
  * 
  * This function takes frames from the queue, performs object tracking,
  * and displays the results.
  * 
  * @param detector Object detector (for class names)
  * @param tracker Object tracker
  * @param frameQueue Queue for receiving frames from the producer
  * @param running Atomic flag indicating if the application is running
  */
 void consumerThread(std::shared_ptr<Detector> detector,
                    std::shared_ptr<KalmanTracker> tracker,
                    ThreadSafeQueue<std::shared_ptr<FrameData>>& frameQueue,
                    std::atomic<bool>& running) {
     // Create viewer
     Viewer viewer("YOLOv8 Object Detection and Tracking", detector->getClassNames());
     
     // Process frames
     while (running) {
         try {
             // Try to get frame data from queue with timeout
             auto optionalFrameData = frameQueue.try_pop_for(std::chrono::milliseconds(100));
             
             if (!optionalFrameData) {
                 // Check if queue is done
                 if (frameQueue.is_done()) {
                     std::cout << "No more frames in queue" << std::endl;
                     break;
                 }
                 continue;
             }
             
             // Unwrap the optional to get the actual shared_ptr
             auto frameDataPtr = *optionalFrameData;
             
             // Create a reference to the FrameData object for cleaner code
             // Note: Not using const here as Viewer::display requires non-const cv::Mat&
             FrameData& data = *frameDataPtr;
             
             // Start measuring processing time
             auto startTime = std::chrono::high_resolution_clock::now();
             
             // Get frame dimensions
             int frameWidth = data.frame.cols;
             int frameHeight = data.frame.rows;
             
             // Update tracker with detections
             std::vector<TrackedObject> trackedObjects = tracker->update(data.detections, frameWidth, frameHeight);
             
             // End measuring processing time
             auto endTime = std::chrono::high_resolution_clock::now();
             float processingTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
             
             // Total processing time (inference + tracking)
             float totalProcessingTime = data.inferenceTime + processingTime;
             
             // Display results
             bool success = viewer.display(
                 data.frame, 
                 data.detections, 
                 trackedObjects, 
                 data.inferenceTime, 
                 totalProcessingTime
             );
             
             // Check if user requested to exit
             if (!success || viewer.isClosed()) {
                 std::cout << "Display closed, exiting" << std::endl;
                 running = false;
                 break;
             }
             
         } catch (const std::exception& e) {
             std::cerr << "Error in consumer thread: " << e.what() << std::endl;
         }
     }
     
     std::cout << "Consumer thread finished" << std::endl;
 }
 
 /**
  * @brief Run occlusion and fast movement tests
  * 
  * This function tests the tracker's performance in challenging scenarios
  * such as occlusion and fast movement. It creates synthetic test cases
  * and measures the tracker's accuracy.
  * 
  * @param detector Object detector
  * @param tracker Object tracker
  */
 void runTests(std::shared_ptr<Detector> detector, std::shared_ptr<KalmanTracker> tracker) {
     std::cout << "\n=== Running test scenarios ===\n";
     
     // Create test frame
     const int width = 800;
     const int height = 600;
     cv::Mat testFrame(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
     
     // Create viewer for test visualization
     Viewer viewer("Test Scenarios", detector->getClassNames());
     
     // Test 1: Occlusion scenario
     std::cout << "Test 1: Occlusion scenario\n";
     
     // Create two objects that will occlude each other
     std::vector<cv::Rect_<float>> objectPaths1, objectPaths2;
     
     // Object 1 moves from left to right
     for (int i = 0; i < 50; i++) {
         objectPaths1.emplace_back(100 + i * 10, 200, 100, 100);
     }
     
     // Object 2 moves from right to left, crossing path with object 1
     for (int i = 0; i < 50; i++) {
         objectPaths2.emplace_back(700 - i * 10, 250, 100, 100);
     }
     
     // Run occlusion test
     for (int frame = 0; frame < 50; frame++) {
         // Clear frame
         testFrame = cv::Scalar(0, 0, 0);
         
         // Create detections
         std::vector<Detection> detections;
         
         // Add object 1 (except during "occlusion")
         if (frame < 20 || frame > 30) {
             // Draw object 1
             cv::rectangle(testFrame, objectPaths1[frame], cv::Scalar(0, 255, 0), -1);
             detections.emplace_back(objectPaths1[frame], 0, 0.9f);
         }
         
         // Add object 2
         cv::rectangle(testFrame, objectPaths2[frame], cv::Scalar(0, 0, 255), -1);
         detections.emplace_back(objectPaths2[frame], 1, 0.9f);
         
         // Update tracker
         std::vector<TrackedObject> trackedObjects = tracker->update(detections, width, height);
         
         // Display results
         viewer.display(testFrame, detections, trackedObjects);
         
         // Slow down for visualization
         cv::waitKey(50);
     }
     
     // Test 2: Fast movement scenario
     std::cout << "Test 2: Fast movement scenario\n";
     
     // Create object with fast movement
     std::vector<cv::Rect_<float>> fastObjectPath;
     
     // Object moves in a zig-zag pattern with varying speeds
     int x = 100, y = 300;
     int dx = 10, dy = 5;
     
     for (int i = 0; i < 100; i++) {
         // Occasionally change direction and speed
         if (i % 10 == 0) {
             dx = (rand() % 30) - 15;
             dy = (rand() % 20) - 10;
         }
         
         // Update position with bounds checking
         x += dx;
         y += dy;
         
         x = std::max(50, std::min(width - 50, x));
         y = std::max(50, std::min(height - 50, y));
         
         fastObjectPath.emplace_back(x - 25, y - 25, 50, 50);
     }
     
     // Run fast movement test
     for (int frame = 0; frame < 100; frame++) {
         // Clear frame
         testFrame = cv::Scalar(0, 0, 0);
         
         // Create detections
         std::vector<Detection> detections;
         
         // Add fast moving object (with occasional misses to simulate detection failures)
         if (frame % 5 != 0) {  // Skip detection every 5th frame
             cv::rectangle(testFrame, fastObjectPath[frame], cv::Scalar(255, 0, 0), -1);
             detections.emplace_back(fastObjectPath[frame], 2, 0.9f);
         }
         
         // Update tracker
         std::vector<TrackedObject> trackedObjects = tracker->update(detections, width, height);
         
         // Display results
         viewer.display(testFrame, detections, trackedObjects);
         
         // Slow down for visualization
         cv::waitKey(30);
     }
     
     std::cout << "Tests completed. Press any key to continue.\n";
     cv::waitKey(0);
 }
 
 /**
  * @brief Main entry point
  * 
  * @param argc Argument count
  * @param argv Argument values
  * @return int Exit code
  */
 int main(int argc, char* argv[]) {
     try {
         // Default parameters
         std::string videoPath;
         std::string modelPath = "models/yolov8s.onnx";
         float confThreshold = 0.25f;
         float nmsThreshold = 0.45f;
         int cameraId = 0;
         bool showDetections = true;
         bool showTracks = true;
         bool showFPS = true;
         
         // Parse command line arguments
         if (!parseArgs(argc, argv, videoPath, modelPath, confThreshold, nmsThreshold,
                       cameraId, showDetections, showTracks, showFPS)) {
             return 1;
         }
         
         // Initialize detector
         std::cout << "Initializing detector with model: " << modelPath << std::endl;
         auto detector = std::make_shared<Detector>(modelPath, confThreshold, nmsThreshold);
         
         // Initialize tracker
         std::cout << "Initializing tracker" << std::endl;
         auto tracker = std::make_shared<KalmanTracker>();
         
         // Open video source
         cv::VideoCapture videoSource;
         if (videoPath.empty()) {
             std::cout << "Opening camera device: " << cameraId << std::endl;
             videoSource.open(cameraId);
         } else {
             std::cout << "Opening video file: " << videoPath << std::endl;
             videoSource.open(videoPath);
         }
         
         if (!videoSource.isOpened()) {
             std::cerr << "Error: Could not open video source" << std::endl;
             return 1;
         }
         
         // Create frame queue for Producer-Consumer pattern
         ThreadSafeQueue<std::shared_ptr<FrameData>> frameQueue;
         
         // Create atomic flag for thread synchronization
         std::atomic<bool> running(true);
         
         // Create and start producer thread
         std::thread producer(producerThread, std::ref(videoSource), detector, std::ref(frameQueue), std::ref(running));
         
         // Create and start consumer thread
         std::thread consumer(consumerThread, detector, tracker, std::ref(frameQueue), std::ref(running));
         
         // Wait for threads to finish
         producer.join();
         consumer.join();
         
         // Close video source
         videoSource.release();
         
         // Run tests if requested
         bool runTestScenarios = false;
         for (int i = 1; i < argc; i++) {
             if (std::string(argv[i]) == "--run-tests") {
                 runTestScenarios = true;
                 break;
             }
         }
         
         if (runTestScenarios) {
             runTests(detector, tracker);
         }
         
         std::cout << "Program completed successfully" << std::endl;
         return 0;
         
     } catch (const std::exception& e) {
         std::cerr << "Error: " << e.what() << std::endl;
         return 1;
     }
 }