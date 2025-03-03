/**
 * @file main.cpp
 * @brief Driver for FastObjectDetection module
 */

#include "FastObjectDetection.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    FastObjectDetection::FmoDetector detector(25, 100, 4.0f, 15);

    cv::Mat frame;
    while (cap.read(frame)) {
        auto fmos = detector.Detect(frame);

        for (const auto& fmo : fmos) {
            cv::rectangle(frame, fmo.BoundingBox, cv::Scalar(0, 255, 0), 2);
            cv::Point center(fmo.BoundingBox.x + fmo.BoundingBox.width / 2,
                             fmo.BoundingBox.y + fmo.BoundingBox.height / 2);
            cv::line(frame, center,
                     cv::Point(center.x + static_cast<int>(fmo.Direction.x * 5),
                               center.y + static_cast<int>(fmo.Direction.y * 5)),
                     cv::Scalar(255, 0, 0), 2);

            std::string speedText = "Speed: " + std::to_string(fmo.Speed);
            cv::putText(frame, speedText,
                        cv::Point(fmo.BoundingBox.x, fmo.BoundingBox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow("FMO Detection", frame);
        if (cv::waitKey(30) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
