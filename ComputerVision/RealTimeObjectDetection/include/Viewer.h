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
     Viewer(const std::string& WindowName, 
            const std::vector<std::string>& ClassNames,
            bool ShowFPS = true);
     
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
     bool Display(cv::Mat& Frame, 
                  const std::vector<Detection>& Detections,
                  const std::vector<TrackedObject>& TrackedObjects,
                  float InferenceTime = 0.0f,
                  float ProcessingTime = 0.0f,
                  cv::Mat* AnnotatedOut = nullptr);
     
     /**
      * @brief Check if the viewer window is closed
      * 
      * @return bool True if the window is closed, false otherwise
      */
     bool IsClosed() const;
     
     /**
      * @brief Set the flag to draw detection boxes
      * 
      * @param draw Whether to draw detection boxes
      */
     void SetDrawDetections(bool Draw) { m_drawDetections = Draw; }
     
     /**
      * @brief Set the flag to draw tracking boxes
      * 
      * @param draw Whether to draw tracking boxes
      */
     void SetDrawTracks(bool Draw) { m_drawTracks = Draw; }
     
     /**
      * @brief Set the flag to show FPS
      * 
      * @param show Whether to show FPS
      */
     void SetShowFPS(bool Show) { m_showFPS = Show; }

     void SetRoiZone(const cv::Rect& Zone) { m_roiZone = Zone; }
     void SetIntrusionAlert(bool Alert) { m_intrusionAlert = Alert; }
 
 private:
     std::string m_windowName;            ///< Name of the display window
     const std::vector<std::string>& m_classNames; ///< Vector of class names
     bool m_drawDetections;               ///< Flag to draw detection boxes
     bool m_drawTracks;                   ///< Flag to draw tracking boxes
     bool m_showFPS;                      ///< Flag to show FPS
     cv::Rect m_roiZone;                  ///< Region of interest for intrusion alert
     bool m_intrusionAlert = false;       ///< Whether intrusion is currently detected
     
     /**
      * @brief Draw detection boxes on the frame
      * 
      * @param frame Frame to draw on
      * @param detections Vector of detections to visualize
      */
     void DrawDetections(cv::Mat& Frame, const std::vector<Detection>& Detections);
     
     /**
      * @brief Draw tracking boxes on the frame
      * 
      * @param frame Frame to draw on
      * @param trackedObjects Vector of tracked objects to visualize
      */
     void DrawTracks(cv::Mat& Frame, const std::vector<TrackedObject>& TrackedObjects);
     
     /**
      * @brief Draw performance metrics on the frame
      * 
      * @param frame Frame to draw on
      * @param inferenceTime Inference time of the detector in milliseconds
      * @param processingTime Total processing time in milliseconds
      */
     void DrawPerformanceMetrics(cv::Mat& Frame, float InferenceTime, float ProcessingTime);

     void DrawRoiZone(cv::Mat& Frame);
 };