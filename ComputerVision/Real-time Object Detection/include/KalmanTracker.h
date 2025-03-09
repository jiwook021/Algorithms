/**
 * @file KalmanTracker.h
 * @brief Object tracker using Kalman Filter
 */

 #pragma once

 #include <memory>
 #include <vector>
 #include <map>
 #include <unordered_map>
 #include <opencv2/opencv.hpp>
 #include "Detector.h"
 
 /**
  * @struct TrackedObject
  * @brief Structure to hold tracked object information
  */
 struct TrackedObject {
     int id;                        ///< Unique ID of the tracked object
     cv::Rect_<float> bbox;         ///< Current bounding box
     cv::Point2f velocity;          ///< Current velocity (pixels/frame)
     int classId;                   ///< Class ID
     float confidence;              ///< Detection confidence
     int age;                       ///< Age in frames
     int totalVisibleCount;         ///< Total number of frames where object was detected
     int consecutiveInvisibleCount; ///< Number of consecutive frames where object was not detected
     cv::Scalar color;              ///< Color for visualization
     std::shared_ptr<cv::KalmanFilter> kalman; ///< Kalman filter for prediction
 
     TrackedObject(int id, const cv::Rect_<float>& bbox, int classId, float conf);
 };
 
 /**
  * @class KalmanTracker
  * @brief Class for tracking objects using Kalman Filter
  * 
  * This class tracks objects detected by the Detector using Kalman Filter.
  * It handles object association, track creation/deletion, and state prediction.
  */
 class KalmanTracker {
 public:
     /**
      * @brief Constructor
      * 
      * @param maxAge Maximum number of frames to keep a track alive without detection
      * @param minHits Minimum number of detections needed to create a track
      * @param iouThreshold IoU threshold for detection-track association
      */
     KalmanTracker(int maxAge = 20, int minHits = 3, float iouThreshold = 0.3f);
 
     /**
      * @brief Update tracks with new detections
      * 
      * @param detections Vector of new detections
      * @param frameWidth Width of the frame
      * @param frameHeight Height of the frame
      * @return std::vector<TrackedObject> Vector of tracked objects
      */
     std::vector<TrackedObject> update(const std::vector<Detection>& detections, 
                                     int frameWidth, int frameHeight);
 
     /**
      * @brief Get the number of active tracks
      * 
      * @return int Number of active tracks
      */
     int getTrackCount() const { return m_tracks.size(); }
 
 private:
     int m_maxAge;                    ///< Maximum number of frames to keep a track alive without detection
     int m_minHits;                   ///< Minimum number of detections needed to create a track
     float m_iouThreshold;            ///< IoU threshold for detection-track association
     int m_nextId;                    ///< Next available track ID
     std::vector<TrackedObject> m_tracks; ///< Vector of tracked objects
     int m_frameCount;                ///< Frame counter
     
     /**
      * @brief Create a new Kalman filter for a track
      * 
      * @param bbox Initial bounding box
      * @return std::shared_ptr<cv::KalmanFilter> Shared pointer to the created Kalman filter
      */
     std::shared_ptr<cv::KalmanFilter> createKalmanFilter(const cv::Rect_<float>& bbox);
     
     /**
      * @brief Calculate IoU (Intersection over Union) between two bounding boxes
      * 
      * @param bb1 First bounding box
      * @param bb2 Second bounding box
      * @return float IoU value
      */
     float calculateIoU(const cv::Rect_<float>& bb1, const cv::Rect_<float>& bb2) const;
     
     /**
      * @brief Associate detections with existing tracks using IoU
      * 
      * @param detections Vector of detections
      * @param cost Matrix for storing cost (1-IoU)
      * @param costThreshold Threshold for valid association
      * @return std::vector<std::pair<int, int>> Vector of (detection index, track index) pairs
      */
     std::vector<std::pair<int, int>> associateDetectionsToTracks(
         const std::vector<Detection>& detections, 
         cv::Mat& cost, 
         float costThreshold) const;
     
     /**
      * @brief Generate a random color for visualization
      * 
      * @return cv::Scalar Random color
      */
     cv::Scalar generateRandomColor() const;
     
     /**
      * @brief Update the Kalman filter with a new detection
      * 
      * @param kalman Kalman filter to update
      * @param bbox Detected bounding box
      */
     void updateKalmanFilter(cv::KalmanFilter* kalman, const cv::Rect_<float>& bbox);
     
     /**
      * @brief Predict the new state using Kalman filter
      * 
      * @param kalman Kalman filter to use for prediction
      * @return cv::Rect_<float> Predicted bounding box
      */
     cv::Rect_<float> predictKalmanFilter(cv::KalmanFilter* kalman);
 };