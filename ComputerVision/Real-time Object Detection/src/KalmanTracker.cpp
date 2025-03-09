/**
 * @file KalmanTracker.cpp
 * @brief Implementation of the KalmanTracker class
 */

 #include "KalmanTracker.h"
 #include <algorithm>
 #include <random>
 #include <iostream>
 
 // Constructor for TrackedObject
 TrackedObject::TrackedObject(int id, const cv::Rect_<float>& bbox, int classId, float conf)
     : id(id),
       bbox(bbox),
       velocity(0, 0),
       classId(classId),
       confidence(conf),
       age(1),
       totalVisibleCount(1),
       consecutiveInvisibleCount(0) {
     // Generate a random color for visualization
     // This is done in the KalmanTracker class
 }
 
 KalmanTracker::KalmanTracker(int maxAge, int minHits, float iouThreshold)
     : m_maxAge(maxAge),
       m_minHits(minHits),
       m_iouThreshold(iouThreshold),
       m_nextId(0),
       m_frameCount(0) {
     // Nothing to initialize here
 }
 
 std::vector<TrackedObject> KalmanTracker::update(const std::vector<Detection>& detections, 
                                                 int frameWidth, int frameHeight) {
     m_frameCount++;
     
     // If no tracks exist and no detections, return empty vector
     if (m_tracks.empty() && detections.empty()) {
         return std::vector<TrackedObject>();
     }
     
     // Predict new locations of existing tracks
     for (auto& track : m_tracks) {
         if (track.kalman) {
             track.bbox = predictKalmanFilter(track.kalman.get());
         }
     }
     
     // Compute cost matrix for detection-track assignment
     cv::Mat cost(detections.size(), m_tracks.size(), CV_32F);
     
     // Fill cost matrix with 1-IoU (so that higher IoU means lower cost)
     for (size_t i = 0; i < detections.size(); i++) {
         for (size_t j = 0; j < m_tracks.size(); j++) {
             float iou = calculateIoU(detections[i].bbox, m_tracks[j].bbox);
             cost.at<float>(i, j) = 1.0f - iou;
         }
     }
     
     // Associate detections to tracks
     float costThreshold = 1.0f - m_iouThreshold; // Convert IoU threshold to cost threshold
     auto assignments = associateDetectionsToTracks(detections, cost, costThreshold);
     
     // Set of unmatched detections and unmatched tracks
     std::vector<bool> detectionMatched(detections.size(), false);
     std::vector<bool> trackMatched(m_tracks.size(), false);
     
     // Update matched tracks with assigned detections
     for (const auto& pair : assignments) {
         int detIdx = pair.first;
         int trkIdx = pair.second;
         
         detectionMatched[detIdx] = true;
         trackMatched[trkIdx] = true;
         
         // Update track with new detection
         m_tracks[trkIdx].bbox = detections[detIdx].bbox;
         m_tracks[trkIdx].confidence = detections[detIdx].confidence;
         m_tracks[trkIdx].age++;
         m_tracks[trkIdx].totalVisibleCount++;
         m_tracks[trkIdx].consecutiveInvisibleCount = 0;
         
         // Update Kalman filter
         updateKalmanFilter(m_tracks[trkIdx].kalman.get(), detections[detIdx].bbox);
     }
     
     // Handle unmatched tracks
     for (size_t i = 0; i < m_tracks.size(); i++) {
         if (!trackMatched[i]) {
             m_tracks[i].age++;
             m_tracks[i].consecutiveInvisibleCount++;
         }
     }
     
     // Handle unmatched detections (create new tracks)
     for (size_t i = 0; i < detections.size(); i++) {
         if (!detectionMatched[i]) {
             TrackedObject newTrack(m_nextId++, detections[i].bbox, detections[i].classId, detections[i].confidence);
             newTrack.kalman = createKalmanFilter(detections[i].bbox);
             newTrack.color = generateRandomColor();
             m_tracks.push_back(newTrack);
         }
     }
     
     // Remove dead tracks
     m_tracks.erase(
         std::remove_if(
             m_tracks.begin(),
             m_tracks.end(),
             [this, frameWidth, frameHeight](const TrackedObject& track) {
                 // Remove if track is too old or went outside frame bounds
                 return track.consecutiveInvisibleCount > m_maxAge ||
                        (track.bbox.x + track.bbox.width <= 0) ||
                        (track.bbox.y + track.bbox.height <= 0) ||
                        (track.bbox.x >= frameWidth) ||
                        (track.bbox.y >= frameHeight);
             }
         ),
         m_tracks.end()
     );
     
     // Return only confirmed tracks (visible for minHits frames)
     std::vector<TrackedObject> result;
     for (const auto& track : m_tracks) {
         if (track.totalVisibleCount >= m_minHits) {
             result.push_back(track);
         }
     }
     
     return result;
 }
 
 std::shared_ptr<cv::KalmanFilter> KalmanTracker::createKalmanFilter(const cv::Rect_<float>& bbox) {
     // Create Kalman filter with state [x, y, width, height, vx, vy, vw, vh]
     // where (x, y) is the center of the box, (width, height) is the size,
     // and (vx, vy, vw, vh) are the respective velocities
     auto kf = std::make_shared<cv::KalmanFilter>(8, 4, 0, CV_32F);
     
     // State transition matrix (F)
     // [1 0 0 0 1 0 0 0]
     // [0 1 0 0 0 1 0 0]
     // [0 0 1 0 0 0 1 0]
     // [0 0 0 1 0 0 0 1]
     // [0 0 0 0 1 0 0 0]
     // [0 0 0 0 0 1 0 0]
     // [0 0 0 0 0 0 1 0]
     // [0 0 0 0 0 0 0 1]
     cv::setIdentity(kf->transitionMatrix);
     for (int i = 0; i < 4; i++) {
         kf->transitionMatrix.at<float>(i, i + 4) = 1.0f; // Add velocity component
     }
     
     // Measurement matrix (H)
     // [1 0 0 0 0 0 0 0]
     // [0 1 0 0 0 0 0 0]
     // [0 0 1 0 0 0 0 0]
     // [0 0 0 1 0 0 0 0]
     kf->measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
     for (int i = 0; i < 4; i++) {
         kf->measurementMatrix.at<float>(i, i) = 1.0f;
     }
     
     // Process noise covariance matrix (Q)
     cv::setIdentity(kf->processNoiseCov, cv::Scalar::all(1e-2));
     
     // Measurement noise covariance matrix (R)
     cv::setIdentity(kf->measurementNoiseCov, cv::Scalar::all(1e-1));
     
     // Posterior error covariance matrix (P)
     cv::setIdentity(kf->errorCovPost, cv::Scalar::all(1.0));
     
     // Initial state (x0)
     kf->statePost.at<float>(0) = bbox.x + bbox.width / 2;  // x center
     kf->statePost.at<float>(1) = bbox.y + bbox.height / 2; // y center
     kf->statePost.at<float>(2) = bbox.width;               // width
     kf->statePost.at<float>(3) = bbox.height;              // height
     kf->statePost.at<float>(4) = 0;                        // vx
     kf->statePost.at<float>(5) = 0;                        // vy
     kf->statePost.at<float>(6) = 0;                        // vw
     kf->statePost.at<float>(7) = 0;                        // vh
     
     return kf;
 }
 
 void KalmanTracker::updateKalmanFilter(cv::KalmanFilter* kf, const cv::Rect_<float>& bbox) {
     // Create measurement vector [x, y, width, height]
     cv::Mat measurement = cv::Mat::zeros(4, 1, CV_32F);
     measurement.at<float>(0) = bbox.x + bbox.width / 2;   // x center
     measurement.at<float>(1) = bbox.y + bbox.height / 2;  // y center
     measurement.at<float>(2) = bbox.width;                // width
     measurement.at<float>(3) = bbox.height;               // height
     
     // Update the Kalman filter with the measurement
     kf->correct(measurement);
 }
 
 cv::Rect_<float> KalmanTracker::predictKalmanFilter(cv::KalmanFilter* kf) {
     // Predict the next state
     cv::Mat prediction = kf->predict();
     
     // Extract the predicted bounding box
     float x_center = prediction.at<float>(0);
     float y_center = prediction.at<float>(1);
     float width = prediction.at<float>(2);
     float height = prediction.at<float>(3);
     
     // Calculate box coordinates
     float x = x_center - width / 2;
     float y = y_center - height / 2;
     
     // Ensure width and height are positive
     width = std::max(width, 1.0f);
     height = std::max(height, 1.0f);
     
     return cv::Rect_<float>(x, y, width, height);
 }
 
 float KalmanTracker::calculateIoU(const cv::Rect_<float>& bb1, const cv::Rect_<float>& bb2) const {
     // Calculate intersection
     float x_left = std::max(bb1.x, bb2.x);
     float y_top = std::max(bb1.y, bb2.y);
     float x_right = std::min(bb1.x + bb1.width, bb2.x + bb2.width);
     float y_bottom = std::min(bb1.y + bb1.height, bb2.y + bb2.height);
     
     if (x_right < x_left || y_bottom < y_top) {
         return 0.0f; // No intersection
     }
     
     float intersection_area = (x_right - x_left) * (y_bottom - y_top);
     
     // Calculate union
     float bb1_area = bb1.width * bb1.height;
     float bb2_area = bb2.width * bb2.height;
     float union_area = bb1_area + bb2_area - intersection_area;
     
     // Calculate IoU
     return intersection_area / union_area;
 }
 
 std::vector<std::pair<int, int>> KalmanTracker::associateDetectionsToTracks(
     const std::vector<Detection>& detections, 
     cv::Mat& cost, 
     float costThreshold) const {
     std::vector<std::pair<int, int>> assignments;
     
     // Use the Hungarian algorithm for assignment
     if (cost.rows > 0 && cost.cols > 0) {
         std::vector<int> assignment;
         
         // Create matrices for Hungarian algorithm
         cv::Mat costMatrix = cost.clone();
         
         // Apply Hungarian algorithm
         // Note: We're using a greedy approach here for simplicity
         while (true) {
             double minVal;
             cv::Point minLoc;
             cv::minMaxLoc(costMatrix, &minVal, nullptr, &minLoc, nullptr);
             
             if (minVal > costThreshold) {
                 break;
             }
             
             int row = minLoc.y;
             int col = minLoc.x;
             
             // Ensure we're within bounds of detections array
             if (row < static_cast<int>(detections.size()) && col < static_cast<int>(m_tracks.size())) {
                 assignments.emplace_back(row, col);
             } else {
                 std::cerr << "Warning: Assignment indices out of bounds: " << row << "," << col << std::endl;
             }
             
             // Mark this row and column as assigned
             for (int i = 0; i < costMatrix.cols; i++) {
                 costMatrix.at<float>(row, i) = FLT_MAX;
             }
             for (int i = 0; i < costMatrix.rows; i++) {
                 costMatrix.at<float>(i, col) = FLT_MAX;
             }
         }
     }
     
     return assignments;
 }
 
 cv::Scalar KalmanTracker::generateRandomColor() const {
     static std::random_device rd;
     static std::mt19937 gen(rd());
     static std::uniform_int_distribution<int> dist(0, 255);
     
     return cv::Scalar(dist(gen), dist(gen), dist(gen));
 }