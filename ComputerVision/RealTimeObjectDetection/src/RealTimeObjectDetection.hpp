/**
 * @file RealTimeObjectDetection.hpp
 * @brief Real-time object detection pipeline with tracking
 * @details Combines YOLO-based detection with lightweight centroid tracking. Maintains object ID persistence across frames, computes FPS, and renders annotated output to display or file.
 */

#pragma once

/**
 * @file main.cpp
 * @brief Main entry point for the object detection and tracking application
 */

 #include <iostream>
 #include <iomanip>
 #include <chrono>
 #include <thread>
 #include <atomic>
 #include <memory>
 #include <filesystem>
 #include <stdexcept>
 #include <opencv2/opencv.hpp>
 
 #include "Detector.h"
 #include "KalmanTracker.h"
 #include "ThreadSafeQueue.h"
 #include "Viewer.h"
 
 namespace Fs = std::filesystem;
 
 /**
  * @struct FrameData
  * @brief Structure to hold frame data for processing
  */

 struct FrameData {
     cv::Mat Frame;               ///< Original frame
     std::vector<Detection> Detections; ///< Detected objects
     int FrameIndex;              ///< Frame index
     float InferenceTime;         ///< Inference time in milliseconds
     
     FrameData(cv::Mat f, int Idx) 
         : Frame(std::move(f)), FrameIndex(Idx), InferenceTime(0.0f) {}
 };
 
 /**
  * @brief Display command line usage information
  * 
  * @param programName Name of the program
  */
 void PrintUsage(const std::string& ProgramName) {
     std::cout << "Usage: " << ProgramName << " [options]\n"
               << "Options:\n"
               << "  --video <path>       : Path to video file (default: use camera)\n"
               << "  --model <path>       : Path to YOLOv8 ONNX model (default: models/yolov8s.onnx)\n"
               << "  --conf <threshold>   : Confidence threshold (default: 0.25)\n"
               << "  --nms <threshold>    : NMS threshold (default: 0.45)\n"
               << "  --camera <id>        : Camera device ID (default: 0)\n"
               << "  --show-detections    : Show raw detections (default: true)\n"
               << "  --show-tracks        : Show tracked objects (default: true)\n"
               << "  --show-fps           : Show FPS and performance metrics (default: true)\n"
               << "  --save-output <path> : Save annotated output video to file\n"
               << "  --roi <x1,y1,x2,y2> : Set intrusion alert zone (pixel coordinates)\n"
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
 bool ParseArgs(int argc, char* argv[],
                std::string& VideoPath,
                std::string& ModelPath,
                float& ConfThreshold,
                float& NmsThreshold,
                int& CameraId,
                bool& ShowDetections,
                bool& ShowTracks,
                bool& ShowFPS,
                std::string& SavePath,
                cv::Rect& RoiZone) {
     for (int i = 1; i < argc; i++) {
         std::string Arg = argv[i];
         
         if (Arg == "--help") {
             PrintUsage(argv[0]);
             return false;
         } else if (Arg == "--video" && i + 1 < argc) {
             VideoPath = argv[++i];
         } else if (Arg == "--model" && i + 1 < argc) {
             ModelPath = argv[++i];
         } else if (Arg == "--conf" && i + 1 < argc) {
             try {
                 ConfThreshold = std::stof(argv[++i]);
                 if (ConfThreshold < 0.0f || ConfThreshold > 1.0f) {
                     std::cerr << "Error: Confidence threshold must be between 0 and 1" << std::endl;
                     return false;
                 }
             } catch (const std::exception& e) {
                 std::cerr << "Error parsing confidence threshold: " << e.what() << std::endl;
                 return false;
             }
         } else if (Arg == "--nms" && i + 1 < argc) {
             try {
                 NmsThreshold = std::stof(argv[++i]);
                 if (NmsThreshold < 0.0f || NmsThreshold > 1.0f) {
                     std::cerr << "Error: NMS threshold must be between 0 and 1" << std::endl;
                     return false;
                 }
             } catch (const std::exception& e) {
                 std::cerr << "Error parsing NMS threshold: " << e.what() << std::endl;
                 return false;
             }
         } else if (Arg == "--camera" && i + 1 < argc) {
             try {
                 CameraId = std::stoi(argv[++i]);
                 if (CameraId < 0) {
                     std::cerr << "Error: Camera ID must be non-negative" << std::endl;
                     return false;
                 }
             } catch (const std::exception& e) {
                 std::cerr << "Error parsing camera ID: " << e.what() << std::endl;
                 return false;
             }
         } else if (Arg == "--show-detections") {
             ShowDetections = true;
         } else if (Arg == "--no-show-detections") {
             ShowDetections = false;
         } else if (Arg == "--show-tracks") {
             ShowTracks = true;
         } else if (Arg == "--no-show-tracks") {
             ShowTracks = false;
         } else if (Arg == "--show-fps") {
             ShowFPS = true;
         } else if (Arg == "--no-show-fps") {
             ShowFPS = false;
         } else if (Arg == "--save-output" && i + 1 < argc) {
             SavePath = argv[++i];
         } else if (Arg == "--roi" && i + 1 < argc) {
             std::string RoiStr = argv[++i];
             int X1, Y1, X2, Y2;
             if (sscanf(RoiStr.c_str(), "%d,%d,%d,%d", &X1, &Y1, &X2, &Y2) == 4) {
                 RoiZone = cv::Rect(cv::Point(std::min(X1, X2), std::min(Y1, Y2)),
                                    cv::Point(std::max(X1, X2), std::max(Y1, Y2)));
             } else {
                 std::cerr << "Error: --roi format must be x1,y1,x2,y2" << std::endl;
                 return false;
             }
         } else if (Arg == "--run-tests") {
             // handled later
         } else {
             std::cerr << "Unknown argument: " << Arg << std::endl;
             PrintUsage(argv[0]);
             return false;
         }
     }
     
     // Validate model path
     if (!Fs::exists(ModelPath)) {
         std::cerr << "Error: Model file not found: " << ModelPath << std::endl;
         return false;
     }
     
     // Validate video path if provided
     if (!VideoPath.empty() && !Fs::exists(VideoPath)) {
         std::cerr << "Error: Video file not found: " << VideoPath << std::endl;
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
 void ProducerThread(cv::VideoCapture& VideoSource,
                   std::shared_ptr<::Detector> DetectorPtr,
                    ThreadSafeQueue<std::shared_ptr<FrameData>>& FrameQueue,
                    std::atomic<bool>& Running) {
     int FrameIndex = 0;
     cv::Mat Frame;

     while (Running) {
         // Read frame
         VideoSource >> Frame;
         
         // Check if frame is valid
         if (Frame.empty()) {
             std::cout << "End of video or failed to read frame" << std::endl;
             break;
         }
         
         // Create frame data
         auto frameData = std::make_shared<FrameData>(Frame.clone(), FrameIndex++);
         
         try {
             // Detect objects
             auto StartTime = std::chrono::high_resolution_clock::now();
             frameData->Detections = DetectorPtr->Detect(frameData->Frame);
             auto EndTime = std::chrono::high_resolution_clock::now();
             
             // Calculate inference time
             frameData->InferenceTime = std::chrono::duration<float, std::milli>(EndTime - StartTime).count();

             // Log detection stats every 30 frames
             if (FrameIndex % 30 == 1) {
                 std::cout << "[Frame " << FrameIndex << "] detections=" << frameData->Detections.size()
                           << "  inference=" << std::fixed << std::setprecision(1)
                           << frameData->InferenceTime << "ms" << std::endl;
             }

             // Push frame data to queue
             FrameQueue.Push(frameData);
             
         } catch (const std::exception& e) {
             std::cerr << "Error in producer thread: " << e.what() << std::endl;
         }
     }
     
     // Signal end of frames
     FrameQueue.Done();
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
void ConsumerThread(std::shared_ptr<::Detector> DetectorPtr,
                   std::shared_ptr<KalmanTracker> Tracker,
                   ThreadSafeQueue<std::shared_ptr<FrameData>>& FrameQueue,
                   std::atomic<bool>& Running,
                   const std::string& SavePath,
                   const cv::Rect& RoiZone) {
     // Create viewer
     ::Viewer viewer("YOLOv8 Object Detection and Tracking", DetectorPtr->GetClassNames());
     if (RoiZone.area() > 0) {
         viewer.SetRoiZone(RoiZone);
     }

     // Optional output video writer
     cv::VideoWriter Writer;
     
     // Process frames
     while (Running) {
         try {
             // Try to get frame data from queue with timeout
             auto OptionalFrameData = FrameQueue.TryPopFor(std::chrono::milliseconds(100));
             
             if (!OptionalFrameData) {
                 // Check if queue is done
                 if (FrameQueue.IsDone()) {
                     std::cout << "No more frames in queue" << std::endl;
                     break;
                 }
                 continue;
             }
             
             // Unwrap the optional to get the actual shared_ptr
             auto FrameDataPtr = *OptionalFrameData;
             
             // Create a reference to the FrameData object for cleaner code
             // Note: Not using const here as Viewer::display requires non-const cv::Mat&
             FrameData& data = *FrameDataPtr;
             
             // Start measuring processing time
             auto StartTime = std::chrono::high_resolution_clock::now();
             
             // Get frame dimensions
             int FrameWidth = data.Frame.cols;
             int FrameHeight = data.Frame.rows;
             
             // Update tracker with detections
             std::vector<TrackedObject> TrackedObjects = Tracker->Update(data.Detections, FrameWidth, FrameHeight);

             // Check for intrusion: person (classId 0) inside ROI zone
             if (RoiZone.area() > 0) {
                 bool Intrusion = false;
                 for (const auto& Obj : TrackedObjects) {
                     if (Obj.ClassId == 0) { // "person"
                         cv::Rect ObjRect(Obj.Bbox);
                         if ((RoiZone & ObjRect).area() > 0) {
                             Intrusion = true;
                             break;
                         }
                     }
                 }
                 viewer.SetIntrusionAlert(Intrusion);
             }

             // End measuring processing time
             auto EndTime = std::chrono::high_resolution_clock::now();
             float ProcessingTime = std::chrono::duration<float, std::milli>(EndTime - StartTime).count();
             
             // Total processing time (inference + tracking)
             float TotalProcessingTime = data.InferenceTime + ProcessingTime;
             
             // Display results (and optionally capture annotated frame for saving)
             cv::Mat AnnotatedFrame;
             bool Success = viewer.Display(
                 data.Frame, 
                 data.Detections, 
                 TrackedObjects, 
                 data.InferenceTime, 
                 TotalProcessingTime,
                 SavePath.empty() ? nullptr : &AnnotatedFrame
             );

             // Write annotated frame to output video if requested
             if (!SavePath.empty() && !AnnotatedFrame.empty()) {
                 if (!Writer.isOpened()) {
                     Writer.open(SavePath,
                                 cv::VideoWriter::fourcc('m','p','4','v'),
                                 30.0,
                                 cv::Size(AnnotatedFrame.cols, AnnotatedFrame.rows));
                 }
                 if (Writer.isOpened()) {
                     Writer.write(AnnotatedFrame);
                 }
             }

             // Check if user requested to exit
             if (!Success || viewer.IsClosed()) {
                 std::cout << "Display closed, exiting" << std::endl;
                 Running = false;
                 break;
             }
             
         } catch (const std::exception& e) {
             std::cerr << "Error in consumer thread: " << e.what() << std::endl;
         }
     }
     
     if (Writer.isOpened()) {
         Writer.release();
         std::cout << "Output video saved to: " << SavePath << std::endl;
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
 void RunTests(std::shared_ptr<::Detector> DetectorPtr, std::shared_ptr<KalmanTracker> Tracker) {
     std::cout << "\n=== Running test scenarios ===\n";
     
     // Create test frame
     const int width = 800;
     const int Height = 600;
     cv::Mat TestFrame(Height, width, CV_8UC3, cv::Scalar(0, 0, 0));
     
     // Create viewer for test visualization
     ::Viewer viewer("Test Scenarios", DetectorPtr->GetClassNames());
     
     // Test 1: Occlusion scenario
     std::cout << "Test 1: Occlusion scenario\n";
     
     // Create two objects that will occlude each other
     std::vector<cv::Rect_<float>> ObjectPaths1, ObjectPaths2;
     
     // Object 1 moves from left to right
     for (int i = 0; i < 50; i++) {
         ObjectPaths1.emplace_back(100 + i * 10, 200, 100, 100);
     }
     
     // Object 2 moves from right to left, crossing path with object 1
     for (int i = 0; i < 50; i++) {
         ObjectPaths2.emplace_back(700 - i * 10, 250, 100, 100);
     }
     
     // Run occlusion test
     for (int Frame = 0; Frame < 50; Frame++) {
         // Clear frame
         TestFrame = cv::Scalar(0, 0, 0);
         
         // Create detections
         std::vector<Detection> Detections;
         
         // Add object 1 (except during "occlusion")
         if (Frame < 20 || Frame > 30) {
             // Draw object 1
             cv::rectangle(TestFrame, ObjectPaths1[Frame], cv::Scalar(0, 255, 0), -1);
             Detections.emplace_back(ObjectPaths1[Frame], 0, 0.9f);
         }
         
         // Add object 2
         cv::rectangle(TestFrame, ObjectPaths2[Frame], cv::Scalar(0, 0, 255), -1);
         Detections.emplace_back(ObjectPaths2[Frame], 1, 0.9f);
         
         // Update tracker
         std::vector<TrackedObject> TrackedObjects = Tracker->Update(Detections, width, Height);
         
         // Display results
         viewer.Display(TestFrame, Detections, TrackedObjects);
         
         // Slow down for visualization
         cv::waitKey(50);
     }
     
     // Test 2: Fast movement scenario
     std::cout << "Test 2: Fast movement scenario\n";
     
     // Create object with fast movement
     std::vector<cv::Rect_<float>> FastObjectPath;
     
     // Object moves in a zig-zag pattern with varying speeds
     int x = 100, y = 300;
     int Dx = 10, Dy = 5;
     
     for (int i = 0; i < 100; i++) {
         // Occasionally change direction and speed
         if (i % 10 == 0) {
             Dx = (rand() % 30) - 15;
             Dy = (rand() % 20) - 10;
         }
         
         // Update position with bounds checking
         x += Dx;
         y += Dy;
         
         x = std::max(50, std::min(width - 50, x));
         y = std::max(50, std::min(Height - 50, y));
         
         FastObjectPath.emplace_back(x - 25, y - 25, 50, 50);
     }
     
     // Run fast movement test
     for (int Frame = 0; Frame < 100; Frame++) {
         // Clear frame
         TestFrame = cv::Scalar(0, 0, 0);
         
         // Create detections
         std::vector<Detection> Detections;
         
         // Add fast moving object (with occasional misses to simulate detection failures)
         if (Frame % 5 != 0) {  // Skip detection every 5th frame
             cv::rectangle(TestFrame, FastObjectPath[Frame], cv::Scalar(255, 0, 0), -1);
             Detections.emplace_back(FastObjectPath[Frame], 2, 0.9f);
         }
         
         // Update tracker
         std::vector<TrackedObject> TrackedObjects = Tracker->Update(Detections, width, Height);
         
         // Display results
         viewer.Display(TestFrame, Detections, TrackedObjects);
         
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
