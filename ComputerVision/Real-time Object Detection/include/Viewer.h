/**
 * @file Viewer.h
 * @brief Viewer class for displaying detection and tracking results
 */

 #pragma once

 #include <string>
 #include <vector>
 #include <memory>
 #include <opencv2/opencv.hpp>
 #include "Detector.h"
 #include "KalmanTracker.h"
 
 /**
  * @class Viewer
  * @brief Class for visualizing detection and tracking results
  * 
  * This class is responsible for displaying the detection and tracking results
  * in an OpenCV window. It provides methods for drawing bounding boxes,
  * tracking information, and performance metrics.
  */
 class Viewer {
 public:
     /**
      * @brief Constructor
      * 
      * @param windowName Name of the display window
      * @param classNames Vector of class names for labeling
      * @param showFPS Whether to show FPS on the display
      */
     Viewer(const std::string& windowName, 
            const std::vector<std::string>& classNames,
            bool showFPS = true);
     
     /**
      * @brief Display a frame with detection and tracking results
      * 
      * @param frame Frame to display
      * @param detections Vector of detections to visualize
      * @param trackedObjects Vector of tracked objects to visualize
      * @param inferenceTime Inference time of the detector in milliseconds
      * @param processingTime Total processing time in milliseconds
      * @return bool True if the frame was displayed successfully, false if user requested to exit
      */
     bool display(cv::Mat& frame, 
                  const std::vector<Detection>& detections,
                  const std::vector<TrackedObject>& trackedObjects,
                  float inferenceTime = 0.0f,
                  float processingTime = 0.0f);
     
     /**
      * @brief Check if the viewer window is closed
      * 
      * @return bool True if the window is closed, false otherwise
      */
     bool isClosed() const;
     
     /**
      * @brief Set the flag to draw detection boxes
      * 
      * @param draw Whether to draw detection boxes
      */
     void setDrawDetections(bool draw) { m_drawDetections = draw; }
     
     /**
      * @brief Set the flag to draw tracking boxes
      * 
      * @param draw Whether to draw tracking boxes
      */
     void setDrawTracks(bool draw) { m_drawTracks = draw; }
     
     /**
      * @brief Set the flag to show FPS
      * 
      * @param show Whether to show FPS
      */
     void setShowFPS(bool show) { m_showFPS = show; }
 
 private:
     std::string m_windowName;            ///< Name of the display window
     const std::vector<std::string>& m_classNames; ///< Vector of class names
     bool m_drawDetections;               ///< Flag to draw detection boxes
     bool m_drawTracks;                   ///< Flag to draw tracking boxes
     bool m_showFPS;                      ///< Flag to show FPS
     
     /**
      * @brief Draw detection boxes on the frame
      * 
      * @param frame Frame to draw on
      * @param detections Vector of detections to visualize
      */
     void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);
     
     /**
      * @brief Draw tracking boxes on the frame
      * 
      * @param frame Frame to draw on
      * @param trackedObjects Vector of tracked objects to visualize
      */
     void drawTracks(cv::Mat& frame, const std::vector<TrackedObject>& trackedObjects);
     
     /**
      * @brief Draw performance metrics on the frame
      * 
      * @param frame Frame to draw on
      * @param inferenceTime Inference time of the detector in milliseconds
      * @param processingTime Total processing time in milliseconds
      */
     void drawPerformanceMetrics(cv::Mat& frame, float inferenceTime, float processingTime);
 };