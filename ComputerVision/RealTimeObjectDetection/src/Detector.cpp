/**
 * @file Detector.cpp
 * @brief YOLOv8 object detector using ONNX Runtime with CUDA Execution Provider
 */

 #include "Detector.h"
 #include <iostream>
 #include <chrono>
 #include <cstring>
 #include <thread>

 Detector::Detector(const std::string& ModelPath, float ConfThreshold, float NmsThreshold)
     : m_env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8Detector"),
       m_confThreshold(ConfThreshold),
       m_nmsThreshold(NmsThreshold),
       m_inferenceTime(0.0f),
       m_inputSize(640, 640),
       m_cudaAvailable(false)
 {
     try {
         Ort::SessionOptions sessionOptions;
         sessionOptions.SetIntraOpNumThreads(1);
         sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

         // Try to add CUDA Execution Provider
         OrtCUDAProviderOptions cudaOptions{};
         cudaOptions.device_id = 0;
         bool cudaEpAdded = false;
         try {
             sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
             cudaEpAdded = true;
         } catch (const Ort::Exception& e) {
             std::cout << "CUDA EP not available: " << e.what() << std::endl;
         }

         // Load model — if CUDA EP causes a device error, retry on CPU
         try {
             m_session = Ort::Session(m_env, ModelPath.c_str(), sessionOptions);
             if (cudaEpAdded) {
                 m_cudaAvailable = true;
                 std::cout << "ONNX Runtime: CUDA Execution Provider active (GPU 0)" << std::endl;
             } else {
                 std::cout << "ONNX Runtime: CPU inference" << std::endl;
             }
         } catch (const Ort::Exception& cudaErr) {
             if (cudaEpAdded) {
                 // CUDA device not accessible (e.g. WSL without /dev/nvidia) — retry CPU
                 std::cout << "CUDA EP failed during session init (" << cudaErr.what()
                           << "), retrying with CPU." << std::endl;
                 Ort::SessionOptions cpuOptions;
                 cpuOptions.SetIntraOpNumThreads(std::max(1u, std::thread::hardware_concurrency() / 2));
                 cpuOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                 m_session = Ort::Session(m_env, ModelPath.c_str(), cpuOptions);
                 m_cudaAvailable = false;
                 std::cout << "ONNX Runtime: CPU inference (fallback)" << std::endl;
             } else {
                 throw;
             }
         }

         // Memory info for CPU input tensors
         m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

         // Collect input/output node names
         Ort::AllocatorWithDefaultOptions allocator;
         size_t numInputs  = m_session.GetInputCount();
         size_t numOutputs = m_session.GetOutputCount();

         for (size_t i = 0; i < numInputs; ++i) {
             auto name = m_session.GetInputNameAllocated(i, allocator);
             m_inputNames.emplace_back(name.get());
         }
         for (size_t i = 0; i < numOutputs; ++i) {
             auto name = m_session.GetOutputNameAllocated(i, allocator);
             m_outputNames.emplace_back(name.get());
         }

         std::cout << "YOLOv8 ONNX model loaded: " << ModelPath
                   << "  inputs=" << numInputs << "  outputs=" << numOutputs << std::endl;

         InitClassNames();

     } catch (const Ort::Exception& e) {
         throw std::runtime_error("ONNX Runtime error loading model: " + std::string(e.what()));
     } catch (const std::exception& e) {
         throw std::runtime_error("Error initializing Detector: " + std::string(e.what()));
     }
 }

 void Detector::InitClassNames() {
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

 std::vector<Detection> Detector::Detect(const cv::Mat& Frame) {
     try {
         if (Frame.empty()) {
             throw std::runtime_error("Empty frame provided to detector");
         }

         auto start = std::chrono::high_resolution_clock::now();

         // Preprocess: resize + BGR→RGB + normalize to float32 NCHW
         cv::Mat resized;
         cv::resize(Frame, resized, m_inputSize);

         cv::Mat floatImg;
         resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);

         // Split BGR channels, store as RGB in NCHW flat blob
         std::vector<cv::Mat> chans(3);
         cv::split(floatImg, chans);  // chans[0]=B [1]=G [2]=R

         const int H = m_inputSize.height;
         const int W = m_inputSize.width;
         std::vector<float> inputBlob(3 * H * W);
         std::memcpy(inputBlob.data() + 0 * H * W, (float*)chans[2].data, H * W * sizeof(float)); // R
         std::memcpy(inputBlob.data() + 1 * H * W, (float*)chans[1].data, H * W * sizeof(float)); // G
         std::memcpy(inputBlob.data() + 2 * H * W, (float*)chans[0].data, H * W * sizeof(float)); // B

         // Create ONNX Runtime input tensor
         std::array<int64_t, 4> inputShape{1, 3, H, W};
         Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
             m_memoryInfo,
             inputBlob.data(), inputBlob.size(),
             inputShape.data(), inputShape.size()
         );

         std::vector<const char*> inNames, outNames;
         for (auto& n : m_inputNames)  inNames.push_back(n.c_str());
         for (auto& n : m_outputNames) outNames.push_back(n.c_str());

         // Run inference
         auto outputs = m_session.Run(
             Ort::RunOptions{nullptr},
             inNames.data(), &inputTensor, 1,
             outNames.data(), outNames.size()
         );

         auto end = std::chrono::high_resolution_clock::now();
         m_inferenceTime = std::chrono::duration<float, std::milli>(end - start).count();

         if (outputs.empty()) return {};

         // Decode output shape
         auto& outTensor = outputs[0];
         auto shape = outTensor.GetTensorTypeAndShapeInfo().GetShape();
         const float* outData = outTensor.GetTensorMutableData<float>();

         // YOLOv8 standard ONNX export: [1, 84, 8400]
         // 84 = 4 (box) + 80 (classes); 8400 candidates
         int rows, cols;
         if (shape.size() == 3) {
             rows = static_cast<int>(shape[1] < shape[2] ? shape[2] : shape[1]);
             cols = static_cast<int>(shape[1] < shape[2] ? shape[1] : shape[2]);
         } else {
             rows = static_cast<int>(outTensor.GetTensorTypeAndShapeInfo().GetElementCount() / 84);
             cols = 84;
         }

         return Postprocess(Frame, outData, rows, cols);

     } catch (const Ort::Exception& e) {
         throw std::runtime_error("ONNX Runtime inference error: " + std::string(e.what()));
     } catch (const std::exception& e) {
         throw std::runtime_error("Error during detection: " + std::string(e.what()));
     }
 }

 std::vector<Detection> Detector::Postprocess(
     const cv::Mat& Frame, const float* outData, int rows, int cols)
 {
     const float xScale = static_cast<float>(Frame.cols) / m_inputSize.width;
     const float yScale = static_cast<float>(Frame.rows) / m_inputSize.height;
     const int numClasses = cols - 4;

     // Transposed layout: [84, 8400] stored as outData[attr * rows + candidate]
     bool transposed = (cols < rows);

     std::vector<Detection> rawDets;
     rawDets.reserve(128);

     for (int i = 0; i < rows; ++i) {
         float cx, cy, w, h;
         int bestClass = -1;
         float bestScore = 0.f;

         if (transposed) {
             cx = outData[0 * rows + i];
             cy = outData[1 * rows + i];
             w  = outData[2 * rows + i];
             h  = outData[3 * rows + i];
             for (int c = 0; c < numClasses; ++c) {
                 float s = outData[(4 + c) * rows + i];
                 if (s > bestScore) { bestScore = s; bestClass = c; }
             }
         } else {
             const float* row = outData + i * cols;
             cx = row[0]; cy = row[1]; w = row[2]; h = row[3];
             for (int c = 0; c < numClasses; ++c) {
                 if (row[4 + c] > bestScore) { bestScore = row[4 + c]; bestClass = c; }
             }
         }

         if (bestScore < m_confThreshold) continue;

         float left   = (cx - w * 0.5f) * xScale;
         float top    = (cy - h * 0.5f) * yScale;
         float right  = (cx + w * 0.5f) * xScale;
         float bottom = (cy + h * 0.5f) * yScale;

         rawDets.emplace_back(
             cv::Rect_<float>(left, top, right - left, bottom - top),
             bestClass, bestScore);
     }

     if (rawDets.empty()) return {};

     std::vector<cv::Rect> boxes;
     std::vector<float>   scores;
     boxes.reserve(rawDets.size());
     scores.reserve(rawDets.size());
     for (auto& d : rawDets) {
         boxes.push_back(cv::Rect(d.Bbox));
         scores.push_back(d.Confidence);
     }

     std::vector<int> indices;
     cv::dnn::NMSBoxes(boxes, scores, m_confThreshold, m_nmsThreshold, indices);

     std::vector<Detection> finalDets;
     finalDets.reserve(indices.size());
     for (int idx : indices) finalDets.push_back(rawDets[idx]);
     return finalDets;
 }
