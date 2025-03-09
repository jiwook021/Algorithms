/**
 * @file Detector.h
 * @brief Object detector using YOLOv8 and OpenCV DNN module with CUDA acceleration
 */

 #pragma once

 #include <string>
 #include <vector>
 #include <memory>
 #include <opencv2/opencv.hpp>
 #include <opencv2/dnn.hpp>
 
 /**
  * @struct Detection
  * @brief Structure to hold detection results including bounding box, class ID, and confidence
  */
 struct Detection {
     cv::Rect_<float> bbox;  ///< Bounding box of the detected object
     int classId;            ///< Class ID of the detected object
     float confidence;       ///< Confidence score of the detection
     
     Detection(const cv::Rect_<float>& box, int cls, float conf)
         : bbox(box), classId(cls), confidence(conf) {}
 };
 
 /**
  * @class Detector
  * @brief Class for object detection using YOLOv8 and OpenCV DNN with CUDA
  * 
  * This class is responsible for loading the YOLOv8 model and detecting objects in frames.
  * It uses OpenCV's DNN module with CUDA acceleration for maximum performance.
  */
 class Detector {
 public:
     /**
      * @brief Constructor
      * 
      * @param modelPath Path to the YOLOv8 ONNX model
      * @param confThreshold Confidence threshold for filtering detections
      * @param nmsThreshold Non-maximum suppression threshold
      * @throw std::runtime_error If the model cannot be loaded or CUDA is not available
      */
     Detector(const std::string& modelPath, float confThreshold = 0.25f, float nmsThreshold = 0.45f);
 
     /**
      * @brief Detect objects in a frame
      * 
      * @param frame Input frame
      * @return std::vector<Detection> Vector of detected objects
      * @throw std::runtime_error If detection fails
      */
     std::vector<Detection> detect(const cv::Mat& frame);
 
     /**
      * @brief Get the list of class names
      * 
      * @return const std::vector<std::string>& Reference to the class names
      */
     const std::vector<std::string>& getClassNames() const { return m_classNames; }
 
     /**
      * @brief Get the input size of the model
      * 
      * @return cv::Size Input size of the model
      */
     cv::Size getInputSize() const { return m_inputSize; }
 
     /**
      * @brief Get the detection time in milliseconds
      * 
      * @return float Detection time in milliseconds
      */
     float getInferenceTime() const { return m_inferenceTime; }
 
 private:
     cv::dnn::Net m_net;                 ///< OpenCV DNN model
     std::vector<std::string> m_classNames; ///< Names of the classes
     cv::Size m_inputSize;               ///< Input size of the model
     float m_confThreshold;              ///< Confidence threshold for filtering detections
     float m_nmsThreshold;               ///< Non-maximum suppression threshold
     float m_inferenceTime;              ///< Detection time in milliseconds
     
     /**
      * @brief Initialize the class names
      */
     void initClassNames();
     
     /**
      * @brief Post-process the network output
      * 
      * @param frame Original frame
      * @param outputs Network outputs
      * @return std::vector<Detection> Vector of detected objects
      */
     std::vector<Detection> postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs);
 };