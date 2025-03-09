/**
 * @file Viewer.cpp
 * @brief Implementation of the Viewer class
 */

 #include "Viewer.h"
 #include <sstream>
 #include <iomanip>
 
 Viewer::Viewer(const std::string& windowName, 
                const std::vector<std::string>& classNames, 
                bool showFPS)
     : m_windowName(windowName),
       m_classNames(classNames),
       m_drawDetections(true),
       m_drawTracks(true),
       m_showFPS(showFPS) {
     // Create the window
     cv::namedWindow(m_windowName, cv::WINDOW_NORMAL);
 }
 
 bool Viewer::display(cv::Mat& frame, 
                     const std::vector<Detection>& detections, 
                     const std::vector<TrackedObject>& trackedObjects, 
                     float inferenceTime, 
                     float processingTime) {
     try {
         // Check if frame is valid
         if (frame.empty()) {
             return false;
         }
         
         // Create a copy of the frame for drawing
         cv::Mat display = frame.clone();
         
         // Draw detections and tracks
         if (m_drawDetections) {
             drawDetections(display, detections);
         }
         
         if (m_drawTracks) {
             drawTracks(display, trackedObjects);
         }
         
         // Draw performance metrics
         if (m_showFPS) {
             drawPerformanceMetrics(display, inferenceTime, processingTime);
         }
         
         // Show the frame
         cv::imshow(m_windowName, display);
         
         // Wait for key press (1ms)
         int key = cv::waitKey(1);
         
         // If ESC key is pressed, return false to signal exit
         if (key == 27) {
             return false;
         }
         
         return true;
         
     } catch (const cv::Exception& e) {
         std::cerr << "OpenCV error in display: " << e.what() << std::endl;
         return false;
     } catch (const std::exception& e) {
         std::cerr << "Error in display: " << e.what() << std::endl;
         return false;
     }
 }
 
 bool Viewer::isClosed() const {
     return !cv::getWindowProperty(m_windowName, cv::WND_PROP_VISIBLE);
 }
 
 void Viewer::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
     for (const auto& det : detections) {
         // Draw bounding box
         cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);
         
         // Prepare label text
         std::string label;
         if (det.classId >= 0 && det.classId < static_cast<int>(m_classNames.size())) {
             label = m_classNames[det.classId];
         } else {
             label = "Unknown";
         }
         label += ": " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
         
         // Draw label background
         int baseLine;
         cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
         cv::rectangle(frame, 
                      cv::Point(det.bbox.x, det.bbox.y - labelSize.height - baseLine - 10),
                      cv::Point(det.bbox.x + labelSize.width, det.bbox.y),
                      cv::Scalar(0, 255, 0), 
                      cv::FILLED);
         
         // Draw label text
         cv::putText(frame, 
                    label, 
                    cv::Point(det.bbox.x, det.bbox.y - baseLine - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    cv::Scalar(0, 0, 0),
                    1,
                    cv::LINE_AA);
     }
 }
 
 void Viewer::drawTracks(cv::Mat& frame, const std::vector<TrackedObject>& trackedObjects) {
     for (const auto& obj : trackedObjects) {
         // Draw bounding box
         cv::rectangle(frame, obj.bbox, obj.color, 2);
         
         // Prepare label text
         std::string label;
         if (obj.classId >= 0 && obj.classId < static_cast<int>(m_classNames.size())) {
             label = m_classNames[obj.classId];
         } else {
             label = "Unknown";
         }
         label += " #" + std::to_string(obj.id);
         
         // Draw label background
         int baseLine;
         cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
         cv::rectangle(frame, 
                      cv::Point(obj.bbox.x, obj.bbox.y - labelSize.height - baseLine - 10),
                      cv::Point(obj.bbox.x + labelSize.width, obj.bbox.y),
                      obj.color, 
                      cv::FILLED);
         
         // Draw label text
         cv::putText(frame, 
                    label, 
                    cv::Point(obj.bbox.x, obj.bbox.y - baseLine - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    cv::Scalar(255, 255, 255),
                    1,
                    cv::LINE_AA);
     }
 }
 
 void Viewer::drawPerformanceMetrics(cv::Mat& frame, float inferenceTime, float processingTime) {
     // Calculate FPS
     float fps = 1000.0f / std::max(1.0f, processingTime);
     
     // Prepare text
     std::stringstream ss;
     ss << std::fixed << std::setprecision(1);
     ss << "Inference: " << inferenceTime << " ms";
     std::string inferenceText = ss.str();
     
     ss.str("");
     ss << "Processing: " << processingTime << " ms";
     std::string processingText = ss.str();
     
     ss.str("");
     ss << "FPS: " << fps;
     std::string fpsText = ss.str();
     
     // Draw background rectangle
     int margin = 10;
     int lineHeight = 20;
     int textHeight = 3 * lineHeight + margin;
     cv::rectangle(frame, 
                  cv::Point(0, 0),
                  cv::Point(200, textHeight),
                  cv::Scalar(0, 0, 0, 0.5), 
                  cv::FILLED);
     
     // Draw text
     cv::putText(frame, 
                inferenceText, 
                cv::Point(margin, margin + lineHeight),
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, 
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
     
     cv::putText(frame, 
                processingText, 
                cv::Point(margin, margin + 2 * lineHeight),
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, 
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
     
     cv::putText(frame, 
                fpsText, 
                cv::Point(margin, margin + 3 * lineHeight),
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, 
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
 }