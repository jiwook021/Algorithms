/**
 * @file Detector.cpp
 * @brief Implementation of the Detector class
 */

 #include "Detector.h"
 #include <iostream>
 #include <fstream>
 #include <chrono>
 
 Detector::Detector(const std::string& modelPath, float confThreshold, float nmsThreshold) 
     : m_confThreshold(confThreshold), m_nmsThreshold(nmsThreshold), m_inferenceTime(0.0f) {
     try {
         // Load the YOLOv8 model
         m_net = cv::dnn::readNetFromONNX(modelPath);
         
         // Set backend and target to CUDA
         m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
         m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
         
         // Check if CUDA is available
         if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
             throw std::runtime_error("CUDA is not available. Please check your OpenCV build.");
         }
         
         // Initialize class names
         initClassNames();
         
         // Set default input size for YOLOv8
         // YOLOv8 models typically use 640x640 input
         m_inputSize = cv::Size(640, 640);
         
         std::cout << "Using default YOLOv8 input size: " << m_inputSize << std::endl;
         
         std::cout << "YOLOv8 model loaded successfully. Input size: " << m_inputSize << std::endl;
     } catch (const cv::Exception& e) {
         throw std::runtime_error("Failed to load YOLOv8 model: " + std::string(e.what()));
     } catch (const std::exception& e) {
         throw std::runtime_error("Error initializing Detector: " + std::string(e.what()));
     }
 }
 
 void Detector::initClassNames() {
     // COCO class names
     m_classNames = {
         "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
         "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
         "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
         "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
         "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
         "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
         "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
         "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
     };
 }
 
 std::vector<Detection> Detector::detect(const cv::Mat& frame) {
     try {
         // Check if frame is empty
         if (frame.empty()) {
             throw std::runtime_error("Empty frame provided to detector");
         }
         
         // Measure inference time
         auto start = std::chrono::high_resolution_clock::now();
         
         // Prepare the input blob - make sure it's the correct format for YOLOv8
         cv::Mat blob;
         
         // Preprocess - resize and normalize
         cv::Mat resized;
         cv::resize(frame, resized, m_inputSize);
         
         // Convert to blob
         blob = cv::dnn::blobFromImage(resized, 1.0/255.0, m_inputSize, cv::Scalar(), true, false);
         
         // Set the input blob
         m_net.setInput(blob);
         
         // Forward pass
         std::vector<cv::Mat> outputs;
         
         // Get output layer names
         std::vector<std::string> outLayerNames = m_net.getUnconnectedOutLayersNames();
         
         // Run forward pass
         m_net.forward(outputs, outLayerNames);
         
         // Calculate inference time
         auto end = std::chrono::high_resolution_clock::now();
         m_inferenceTime = std::chrono::duration<float, std::milli>(end - start).count();
         
         // Handle different output formats
         if (outputs.empty()) {
             std::cout << "Warning: No outputs from the network" << std::endl;
             return {};
         }
         
         std::cout << "Network output shape: " << outputs[0].dims << " dimensions" << std::endl;
         for (int i = 0; i < outputs[0].dims; i++) {
             std::cout << "Dim " << i << ": " << outputs[0].size[i] << std::endl;
         }
         
         // Postprocess the outputs
         return postprocess(frame, outputs);
         
     } catch (const cv::Exception& e) {
         throw std::runtime_error("OpenCV error during detection: " + std::string(e.what()));
     } catch (const std::exception& e) {
         throw std::runtime_error("Error during detection: " + std::string(e.what()));
     }
 }
 
 std::vector<Detection> Detector::postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs) {
     std::vector<Detection> detections;
     
     // Check for empty outputs
     if (outputs.empty()) {
         std::cerr << "Error: Empty network output" << std::endl;
         return detections;
     }
     
     // YOLOv8 has a different output format compared to YOLOv5/v7
     // Let's handle different possible formats
     const cv::Mat& output = outputs[0];
     
     // Check if output is valid
     if (output.empty()) {
         std::cerr << "Error: Invalid output (empty)" << std::endl;
         return detections;
     }
     
     // Scaling factors for bounding box
     float xFactor = static_cast<float>(frame.cols) / m_inputSize.width;
     float yFactor = static_cast<float>(frame.rows) / m_inputSize.height;
     
     // Check output format (YOLOv8 changes the output format compared to v5/v7)
     // Format 1: [num_detections, 85] where 85 = 4 (box) + 1 (conf) + 80 (classes)
     // Format 2: [1, 25200, 85] where 25200 is num of detected objects
     
     // Adapt to the output format
     int rows = output.rows;
     int dimensions = output.cols;
     
     // Debug output format
     std::cout << "YOLO output shape: " << output.rows << "x" << output.cols << std::endl;
     
     // Process each detection
     for (int i = 0; i < rows; i++) {
         // Access row data (works for both output formats)
         const float* row_ptr = output.ptr<float>(i);
         
         // For YOLOv8 with format [num_detections, 4+num_classes]
         // The first 4 elements are x, y, w, h
         float x = row_ptr[0];
         float y = row_ptr[1];
         float width = row_ptr[2];
         float height = row_ptr[3];
         
         // Find class with highest score
         int classId = -1;
         float maxScore = -1;
         
         // Start from index 4 through the rest of the columns
         // This varies based on YOLOv8 output format, but generally it's [box(4), confidence(1), classes(80)]
         // or just [box(4), classes(80)] with confidence embedded in class scores
         
         int scoreOffset = 4;  // Default to having confidence at index 4
         
         // Check if we have a confidence score separate from classes
         if (dimensions > 85) {  // Old format with separate confidence
             float confidence = row_ptr[4];
             scoreOffset = 5;  // Class scores start at index 5
             
             // If confidence is too low, skip this detection
             if (confidence < m_confThreshold) {
                 continue;
             }
         }
         
         // Find class with highest score
         for (int j = scoreOffset; j < dimensions; j++) {
             float score = row_ptr[j];
             if (score > maxScore) {
                 maxScore = score;
                 classId = j - scoreOffset;  // Adjust index to start from 0
             }
         }
         
         // Filter by confidence threshold
         if (maxScore >= m_confThreshold) {
             // Convert to top-left coordinates and scale to original image size
             float left = (x - width/2) * xFactor;
             float top = (y - height/2) * yFactor;
             float right = (x + width/2) * xFactor;
             float bottom = (y + height/2) * yFactor;
             
             // Create bounding box
             cv::Rect_<float> bbox(left, top, right - left, bottom - top);
             
             detections.emplace_back(bbox, classId, maxScore);
         }
     }
     
     // Non-maximum suppression to remove overlapping boxes
     std::vector<int> indices;
     std::vector<cv::Rect> boxes;
     std::vector<float> scores;
     
     for (const auto& detection : detections) {
         boxes.push_back(cv::Rect(detection.bbox));
         scores.push_back(detection.confidence);
     }
     
     if (!boxes.empty()) {
         // Apply NMS
         cv::dnn::NMSBoxes(boxes, scores, m_confThreshold, m_nmsThreshold, indices);
         
         // Create final detections
         std::vector<Detection> finalDetections;
         for (int idx : indices) {
             finalDetections.push_back(detections[idx]);
         }
         
         return finalDetections;
     }
     
     return detections;  // If no boxes, return original detections (empty vector)
 }