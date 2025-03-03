/**
 * @file Detector.h
 * @brief Object detector using YOLOv8 and ONNX Runtime with CUDA acceleration
 */

 #pragma once

 #include <string>
 #include <vector>
 #include <memory>
 #include <opencv2/opencv.hpp>
 #include <onnxruntime_cxx_api.h>
 
 /**
  * @struct Detection
  * @brief Structure to hold detection results including bounding box, class ID, and confidence
  */
 struct Detection {
     cv::Rect_<float> Bbox;  ///< Bounding box of the detected object
     int ClassId;            ///< Class ID of the detected object
     float Confidence;       ///< Confidence score of the detection
     
     Detection(const cv::Rect_<float>& Box, int Cls, float Conf)
         : Bbox(Box), ClassId(Cls), Confidence(Conf) {}
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
     Detector(const std::string& ModelPath, float ConfThreshold = 0.25f, float NmsThreshold = 0.45f);
 
     /**
      * @brief Detect objects in a frame
      * 
      * @param frame Input frame
      * @return std::vector<Detection> Vector of detected objects
      * @throw std::runtime_error If detection fails
      */
     std::vector<Detection> Detect(const cv::Mat& Frame);
 
     /**
      * @brief Get the list of class names
      * 
      * @return const std::vector<std::string>& Reference to the class names
      */
     const std::vector<std::string>& GetClassNames() const { return m_classNames; }
 
     /**
      * @brief Get the input size of the model
      * 
      * @return cv::Size Input size of the model
      */
     cv::Size GetInputSize() const { return m_inputSize; }
 
     /**
      * @brief Get the detection time in milliseconds
      * 
      * @return float Detection time in milliseconds
      */
     float GetInferenceTime() const { return m_inferenceTime; }
 
 private:
     Ort::Env m_env;                              ///< ONNX Runtime environment
     Ort::Session m_session{nullptr};             ///< ONNX Runtime inference session
     Ort::MemoryInfo m_memoryInfo{nullptr};       ///< Memory info for input tensors
     std::vector<std::string> m_classNames;       ///< Names of the classes
     std::vector<std::string> m_inputNames;       ///< Model input node names
     std::vector<std::string> m_outputNames;      ///< Model output node names
     cv::Size m_inputSize;                        ///< Input size of the model
     float m_confThreshold;                       ///< Confidence threshold for filtering detections
     float m_nmsThreshold;                        ///< Non-maximum suppression threshold
     float m_inferenceTime;                       ///< Detection time in milliseconds
     bool m_cudaAvailable;                        ///< Whether CUDA was successfully initialized

     /**
      * @brief Initialize the class names
      */
     void InitClassNames();

     /**
      * @brief Post-process the ONNX Runtime output
      *
      * @param frame Original frame
      * @param output Raw float output from the model [1, 84, 8400] or similar
      * @param rows Number of output rows
      * @param cols Number of output cols
      * @return std::vector<Detection> Vector of detected objects
      */
     std::vector<Detection> Postprocess(const cv::Mat& Frame, const float* output, int rows, int cols);
 };