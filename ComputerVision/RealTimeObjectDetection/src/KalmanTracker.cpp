/**
 * @file KalmanTracker.cpp
 * @brief Implementation of the KalmanTracker class
 */

 #include "KalmanTracker.h"
 #include <algorithm>
 #include <random>
 #include <iostream>
 
 // Constructor for TrackedObject
 TrackedObject::TrackedObject(int Id, const cv::Rect_<float>& Bbox, int ClassId, float Conf)
     : Id(Id),
       Bbox(Bbox),
       Velocity(0, 0),
       ClassId(ClassId),
       Confidence(Conf),
       Age(1),
       TotalVisibleCount(1),
       ConsecutiveInvisibleCount(0) {
     // Generate a random color for visualization
     // This is done in the KalmanTracker class
 }
 
 KalmanTracker::KalmanTracker(int MaxAge, int MinHits, float IouThreshold)
     : m_maxAge(MaxAge),
       m_minHits(MinHits),
       m_iouThreshold(IouThreshold),
       m_nextId(0),
       m_frameCount(0) {
     // Nothing to initialize here
 }
 
 std::vector<TrackedObject> KalmanTracker::Update(const std::vector<Detection>& Detections, 
                                                 int FrameWidth, int FrameHeight) {
     m_frameCount++;
     
     // If no tracks exist and no detections, return empty vector
     if (MTracks.empty() && Detections.empty()) {
         return std::vector<TrackedObject>();
     }
     
     // Predict new locations of existing tracks
     for (auto& Track : MTracks) {
         if (Track.Kalman) {
             Track.Bbox = PredictKalmanFilter(Track.Kalman.get());
         }
     }
     
     // Compute cost matrix for detection-track assignment
     cv::Mat Cost(Detections.size(), MTracks.size(), CV_32F);
     
     // Fill cost matrix with 1-IoU (so that higher IoU means lower cost)
     for (size_t i = 0; i < Detections.size(); i++) {
         for (size_t j = 0; j < MTracks.size(); j++) {
             float Iou = CalculateIoU(Detections[i].Bbox, MTracks[j].Bbox);
             Cost.at<float>(i, j) = 1.0f - Iou;
         }
     }
     
     // Associate detections to tracks
     float CostThreshold = 1.0f - m_iouThreshold; // Convert IoU threshold to cost threshold
     auto Assignments = AssociateDetectionsToTracks(Detections, Cost, CostThreshold);
     
     // Set of unmatched detections and unmatched tracks
     std::vector<bool> DetectionMatched(Detections.size(), false);
     std::vector<bool> TrackMatched(MTracks.size(), false);
     
     // Update matched tracks with assigned detections
     for (const auto& pair : Assignments) {
         int DetIdx = pair.first;
         int TrkIdx = pair.second;
         
         DetectionMatched[DetIdx] = true;
         TrackMatched[TrkIdx] = true;
         
         // Update track with new detection
         MTracks[TrkIdx].Bbox = Detections[DetIdx].Bbox;
         MTracks[TrkIdx].Confidence = Detections[DetIdx].Confidence;
         MTracks[TrkIdx].Age++;
         MTracks[TrkIdx].TotalVisibleCount++;
         MTracks[TrkIdx].ConsecutiveInvisibleCount = 0;
         
         // Update Kalman filter
         UpdateKalmanFilter(MTracks[TrkIdx].Kalman.get(), Detections[DetIdx].Bbox);
     }
     
     // Handle unmatched tracks
     for (size_t i = 0; i < MTracks.size(); i++) {
         if (!TrackMatched[i]) {
             MTracks[i].Age++;
             MTracks[i].ConsecutiveInvisibleCount++;
         }
     }
     
     // Handle unmatched detections (create new tracks)
     for (size_t i = 0; i < Detections.size(); i++) {
         if (!DetectionMatched[i]) {
             TrackedObject NewTrack(m_nextId++, Detections[i].Bbox, Detections[i].ClassId, Detections[i].Confidence);
             NewTrack.Kalman = CreateKalmanFilter(Detections[i].Bbox);
             NewTrack.Color = GenerateRandomColor();
             MTracks.push_back(NewTrack);
         }
     }
     
     // Remove dead tracks
     MTracks.erase(
         std::remove_if(
             MTracks.begin(),
             MTracks.end(),
             [this, FrameWidth, FrameHeight](const TrackedObject& Track) {
                 // Remove if track is too old or went outside frame bounds
                 return Track.ConsecutiveInvisibleCount > m_maxAge ||
                        (Track.Bbox.x + Track.Bbox.width <= 0) ||
                        (Track.Bbox.y + Track.Bbox.height <= 0) ||
                        (Track.Bbox.x >= FrameWidth) ||
                        (Track.Bbox.y >= FrameHeight);
             }
         ),
         MTracks.end()
     );
     
     // Return only confirmed tracks (visible for minHits frames)
     std::vector<TrackedObject> Result;
     for (const auto& Track : MTracks) {
         if (Track.TotalVisibleCount >= m_minHits) {
             Result.push_back(Track);
         }
     }
     
     return Result;
 }
 
 std::shared_ptr<cv::KalmanFilter> KalmanTracker::CreateKalmanFilter(const cv::Rect_<float>& Bbox) {
     // Create Kalman filter with state [x, y, width, height, vx, vy, vw, vh]
     // where (x, y) is the center of the box, (width, height) is the size,
     // and (vx, vy, vw, vh) are the respective velocities
     auto Kf = std::make_shared<cv::KalmanFilter>(8, 4, 0, CV_32F);
     
     // State transition matrix (F)
     // [1 0 0 0 1 0 0 0]
     // [0 1 0 0 0 1 0 0]
     // [0 0 1 0 0 0 1 0]
     // [0 0 0 1 0 0 0 1]
     // [0 0 0 0 1 0 0 0]
     // [0 0 0 0 0 1 0 0]
     // [0 0 0 0 0 0 1 0]
     // [0 0 0 0 0 0 0 1]
     cv::setIdentity(Kf->transitionMatrix);
     for (int i = 0; i < 4; i++) {
         Kf->transitionMatrix.at<float>(i, i + 4) = 1.0f; // Add velocity component
     }
     
     // Measurement matrix (H)
     // [1 0 0 0 0 0 0 0]
     // [0 1 0 0 0 0 0 0]
     // [0 0 1 0 0 0 0 0]
     // [0 0 0 1 0 0 0 0]
     Kf->measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
     for (int i = 0; i < 4; i++) {
         Kf->measurementMatrix.at<float>(i, i) = 1.0f;
     }
     
     // Process noise covariance matrix (Q)
     cv::setIdentity(Kf->processNoiseCov, cv::Scalar::all(1e-2));
     
     // Measurement noise covariance matrix (R)
     cv::setIdentity(Kf->measurementNoiseCov, cv::Scalar::all(1e-1));
     
     // Posterior error covariance matrix (P)
     cv::setIdentity(Kf->errorCovPost, cv::Scalar::all(1.0));
     
     // Initial state (x0)
     Kf->statePost.at<float>(0) = Bbox.x + Bbox.width / 2;  // x center
     Kf->statePost.at<float>(1) = Bbox.y + Bbox.height / 2; // y center
     Kf->statePost.at<float>(2) = Bbox.width;               // width
     Kf->statePost.at<float>(3) = Bbox.height;              // height
     Kf->statePost.at<float>(4) = 0;                        // vx
     Kf->statePost.at<float>(5) = 0;                        // vy
     Kf->statePost.at<float>(6) = 0;                        // vw
     Kf->statePost.at<float>(7) = 0;                        // vh
     
     return Kf;
 }
 
 void KalmanTracker::UpdateKalmanFilter(cv::KalmanFilter* Kf, const cv::Rect_<float>& Bbox) {
     // Create measurement vector [x, y, width, height]
     cv::Mat Measurement = cv::Mat::zeros(4, 1, CV_32F);
     Measurement.at<float>(0) = Bbox.x + Bbox.width / 2;   // x center
     Measurement.at<float>(1) = Bbox.y + Bbox.height / 2;  // y center
     Measurement.at<float>(2) = Bbox.width;                // width
     Measurement.at<float>(3) = Bbox.height;               // height
     
     // Update the Kalman filter with the measurement
     Kf->correct(Measurement);
 }
 
 cv::Rect_<float> KalmanTracker::PredictKalmanFilter(cv::KalmanFilter* Kf) {
     // Predict the next state
     cv::Mat Prediction = Kf->predict();
     
     // Extract the predicted bounding box
     float XCenter = Prediction.at<float>(0);
     float YCenter = Prediction.at<float>(1);
     float width = Prediction.at<float>(2);
     float Height = Prediction.at<float>(3);
     
     // Calculate box coordinates
     float x = XCenter - width / 2;
     float y = YCenter - Height / 2;
     
     // Ensure width and height are positive
     width = std::max(width, 1.0f);
     Height = std::max(Height, 1.0f);
     
     return cv::Rect_<float>(x, y, width, Height);
 }
 
 float KalmanTracker::CalculateIoU(const cv::Rect_<float>& Bb1, const cv::Rect_<float>& Bb2) const {
     // Calculate intersection
     float XLeft = std::max(Bb1.x, Bb2.x);
     float YTop = std::max(Bb1.y, Bb2.y);
     float XRight = std::min(Bb1.x + Bb1.width, Bb2.x + Bb2.width);
     float YBottom = std::min(Bb1.y + Bb1.height, Bb2.y + Bb2.height);
     
     if (XRight < XLeft || YBottom < YTop) {
         return 0.0f; // No intersection
     }
     
     float IntersectionArea = (XRight - XLeft) * (YBottom - YTop);
     
     // Calculate union
     float Bb1Area = Bb1.width * Bb1.height;
     float Bb2Area = Bb2.width * Bb2.height;
     float UnionArea = Bb1Area + Bb2Area - IntersectionArea;
     
     // Calculate IoU
     return IntersectionArea / UnionArea;
 }
 
 std::vector<std::pair<int, int>> KalmanTracker::AssociateDetectionsToTracks(
     const std::vector<Detection>& Detections, 
     cv::Mat& Cost, 
     float CostThreshold) const {
     std::vector<std::pair<int, int>> Assignments;
     
     // Use the Hungarian algorithm for assignment
     if (Cost.rows > 0 && Cost.cols > 0) {
         std::vector<int> Assignment;
         
         // Create matrices for Hungarian algorithm
         cv::Mat CostMatrix = Cost.clone();
         
         // Apply Hungarian algorithm
         // Note: We're using a greedy approach here for simplicity
         while (true) {
             double MinVal;
             cv::Point MinLoc;
             cv::minMaxLoc(CostMatrix, &MinVal, nullptr, &MinLoc, nullptr);
             
             if (MinVal > CostThreshold) {
                 break;
             }
             
             int Row = MinLoc.y;
             int Col = MinLoc.x;
             
             // Ensure we're within bounds of detections array
             if (Row < static_cast<int>(Detections.size()) && Col < static_cast<int>(MTracks.size())) {
                 Assignments.emplace_back(Row, Col);
             } else {
                 std::cerr << "Warning: Assignment indices out of bounds: " << Row << "," << Col << std::endl;
             }
             
             // Mark this row and column as assigned
             for (int i = 0; i < CostMatrix.cols; i++) {
                 CostMatrix.at<float>(Row, i) = FLT_MAX;
             }
             for (int i = 0; i < CostMatrix.rows; i++) {
                 CostMatrix.at<float>(i, Col) = FLT_MAX;
             }
         }
     }
     
     return Assignments;
 }
 
 cv::Scalar KalmanTracker::GenerateRandomColor() const {
     static std::random_device Rd;
     static std::mt19937 Gen(Rd());
     static std::uniform_int_distribution<int> Dist(0, 255);
     
     return cv::Scalar(Dist(Gen), Dist(Gen), Dist(Gen));
 }