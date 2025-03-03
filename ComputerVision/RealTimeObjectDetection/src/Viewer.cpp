/**
 * @file Viewer.cpp
 * @brief Implementation of the Viewer class
 */

 #include "Viewer.h"
 #include <sstream>
 #include <iomanip>
 
 Viewer::Viewer(const std::string& WindowName, 
                const std::vector<std::string>& ClassNames, 
                bool ShowFPS)
     : m_windowName(WindowName),
       m_classNames(ClassNames),
       m_drawDetections(true),
       m_drawTracks(true),
       m_showFPS(ShowFPS) {
     // Create the window
     cv::namedWindow(m_windowName, cv::WINDOW_NORMAL);
 }
 
 bool Viewer::Display(cv::Mat& Frame, 
                     const std::vector<Detection>& Detections, 
                     const std::vector<TrackedObject>& TrackedObjects, 
                     float InferenceTime, 
                     float ProcessingTime,
                     cv::Mat* AnnotatedOut) {
     try {
         // Check if frame is valid
         if (Frame.empty()) {
             return false;
         }
         
         // Create a copy of the frame for drawing
         cv::Mat Display = Frame.clone();
         
         // Draw detections and tracks
         if (m_drawDetections) {
             DrawDetections(Display, Detections);
         }
         
         if (m_drawTracks) {
             DrawTracks(Display, TrackedObjects);
         }
         
         // Draw ROI zone and intrusion alert
         if (m_roiZone.area() > 0) {
             DrawRoiZone(Display);
         }

         // Draw performance metrics
         if (m_showFPS) {
             DrawPerformanceMetrics(Display, InferenceTime, ProcessingTime);
         }

         // Optionally copy annotated frame to caller's buffer
         if (AnnotatedOut != nullptr) {
             *AnnotatedOut = Display.clone();
         }

         // Show the frame
         cv::imshow(m_windowName, Display);
         
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
 
 bool Viewer::IsClosed() const {
     return !cv::getWindowProperty(m_windowName, cv::WND_PROP_VISIBLE);
 }
 
 void Viewer::DrawDetections(cv::Mat& Frame, const std::vector<Detection>& Detections) {
     for (const auto& Det : Detections) {
         // Draw bounding box
         cv::rectangle(Frame, Det.Bbox, cv::Scalar(0, 255, 0), 2);
         
         // Prepare label text
         std::string Label;
         if (Det.ClassId >= 0 && Det.ClassId < static_cast<int>(m_classNames.size())) {
             Label = m_classNames[Det.ClassId];
         } else {
             Label = "Unknown";
         }
         Label += ": " + std::to_string(static_cast<int>(Det.Confidence * 100)) + "%";
         
         // Draw label background
         int BaseLine;
         cv::Size LabelSize = cv::getTextSize(Label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &BaseLine);
         cv::rectangle(Frame, 
                      cv::Point(Det.Bbox.x, Det.Bbox.y - LabelSize.height - BaseLine - 10),
                      cv::Point(Det.Bbox.x + LabelSize.width, Det.Bbox.y),
                      cv::Scalar(0, 255, 0), 
                      cv::FILLED);
         
         // Draw label text
         cv::putText(Frame, 
                    Label, 
                    cv::Point(Det.Bbox.x, Det.Bbox.y - BaseLine - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    cv::Scalar(0, 0, 0),
                    1,
                    cv::LINE_AA);
     }
 }
 
 void Viewer::DrawTracks(cv::Mat& Frame, const std::vector<TrackedObject>& TrackedObjects) {
     for (const auto& Obj : TrackedObjects) {
         // Draw bounding box
         cv::rectangle(Frame, Obj.Bbox, Obj.Color, 2);
         
         // Prepare label text
         std::string Label;
         if (Obj.ClassId >= 0 && Obj.ClassId < static_cast<int>(m_classNames.size())) {
             Label = m_classNames[Obj.ClassId];
         } else {
             Label = "Unknown";
         }
         Label += " #" + std::to_string(Obj.Id);
         
         // Draw label background
         int BaseLine;
         cv::Size LabelSize = cv::getTextSize(Label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &BaseLine);
         cv::rectangle(Frame, 
                      cv::Point(Obj.Bbox.x, Obj.Bbox.y - LabelSize.height - BaseLine - 10),
                      cv::Point(Obj.Bbox.x + LabelSize.width, Obj.Bbox.y),
                      Obj.Color, 
                      cv::FILLED);
         
         // Draw label text
         cv::putText(Frame, 
                    Label, 
                    cv::Point(Obj.Bbox.x, Obj.Bbox.y - BaseLine - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    cv::Scalar(255, 255, 255),
                    1,
                    cv::LINE_AA);
     }
 }
 
 void Viewer::DrawPerformanceMetrics(cv::Mat& Frame, float InferenceTime, float ProcessingTime) {
     // Calculate FPS
     float Fps = 1000.0f / std::max(1.0f, ProcessingTime);
     
     // Prepare text
     std::stringstream Ss;
     Ss << std::fixed << std::setprecision(1);
     Ss << "Inference: " << InferenceTime << " ms";
     std::string InferenceText = Ss.str();
     
     Ss.str("");
     Ss << "Processing: " << ProcessingTime << " ms";
     std::string ProcessingText = Ss.str();
     
     Ss.str("");
     Ss << "FPS: " << Fps;
     std::string FpsText = Ss.str();
     
     // Draw background rectangle
     int Margin = 10;
     int LineHeight = 20;
     int TextHeight = 3 * LineHeight + Margin;
     cv::rectangle(Frame, 
                  cv::Point(0, 0),
                  cv::Point(200, TextHeight),
                  cv::Scalar(0, 0, 0, 0.5), 
                  cv::FILLED);
     
     // Draw text
     cv::putText(Frame, 
                InferenceText, 
                cv::Point(Margin, Margin + LineHeight),
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, 
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
     
     cv::putText(Frame, 
                ProcessingText, 
                cv::Point(Margin, Margin + 2 * LineHeight),
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, 
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
     
     cv::putText(Frame,
                FpsText, 
                cv::Point(Margin, Margin + 3 * LineHeight),
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, 
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);
 }

 void Viewer::DrawRoiZone(cv::Mat& Frame) {
     // Draw the ROI boundary — green when clear, red when intrusion detected
     cv::Scalar ZoneColor = m_intrusionAlert ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
     int Thickness = m_intrusionAlert ? 3 : 2;
     cv::rectangle(Frame, m_roiZone, ZoneColor, Thickness);

     // Semi-transparent overlay on the zone
     cv::Mat Overlay = Frame.clone();
     cv::rectangle(Overlay, m_roiZone, ZoneColor, cv::FILLED);
     double Alpha = m_intrusionAlert ? 0.25 : 0.08;
     cv::addWeighted(Overlay, Alpha, Frame, 1.0 - Alpha, 0, Frame);

     // Zone label
     std::string ZoneLabel = m_intrusionAlert ? "!! INTRUSION ALERT !!" : "Monitoring Zone";
     int BaseLine;
     cv::Size LabelSize = cv::getTextSize(ZoneLabel, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &BaseLine);

     cv::Point LabelPos(m_roiZone.x, m_roiZone.y - 10);
     if (LabelPos.y - LabelSize.height < 0) {
         LabelPos.y = m_roiZone.y + LabelSize.height + 10;
     }

     // Label background
     cv::rectangle(Frame,
                  cv::Point(LabelPos.x - 2, LabelPos.y - LabelSize.height - 4),
                  cv::Point(LabelPos.x + LabelSize.width + 2, LabelPos.y + 4),
                  ZoneColor, cv::FILLED);

     cv::putText(Frame, ZoneLabel, LabelPos,
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

     // Flashing banner across the top when alert is active
     if (m_intrusionAlert) {
         int BannerH = 40;
         cv::Mat Banner = Frame(cv::Rect(0, 0, Frame.cols, BannerH));
         Banner.setTo(cv::Scalar(0, 0, 200));
         cv::addWeighted(Banner, 0.7, Frame(cv::Rect(0, 0, Frame.cols, BannerH)), 0.3, 0, Banner);

         cv::putText(Frame, "PERSON DETECTED IN RESTRICTED ZONE",
                    cv::Point(Frame.cols / 2 - 250, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
     }
 }