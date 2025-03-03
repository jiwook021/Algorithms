/**
 * @file main.cpp
 * @brief Entry point for the MotionDetector module.
 *
 * Processes a video file for motion detection, logging events to SQLite
 * and saving frames as images.
 */

#include "MotionDetector.hpp"
#include <iostream>
#include <string>
#include <cstdlib>

/**
 * @brief Main entry point.
 *
 * Usage: ./main [video_path]
 * If no path is provided, defaults to ~/Algorithms/data/video.mp4.
 *
 * @param argc Argument count.
 * @param argv Argument values.
 * @return 0 on success, 1 on failure.
 */
int main(int argc, char* argv[]) {
    std::string videoPath;
    if (argc >= 2) {
        videoPath = argv[1];
    } else {
        const char* home = std::getenv("HOME");
        videoPath = std::string(home ? home : ".") +
                    "/Algorithms/data/video.mp4";
    }

    std::cout << "Motion detection from video: " << videoPath << "\n";

    MotionDetector detector("motion_log.db");

    if (!detector.InitSchema()) {
        std::cerr << "Failed to initialize database schema.\n";
        return 1;
    }

    int eventCount = detector.ProcessVideo(videoPath, true);
    if (eventCount < 0) {
        std::cerr << "Failed to process video.\n";
        return 1;
    }

    std::cout << "Detected " << eventCount << " motion events.\n";

    // Display all logged events
    auto events = detector.GetAllEvents();
    std::cout << "\nAll logged events (" << events.size() << "):\n";
    for (const auto& event : events) {
        std::cout << "  " << event.ToString() << "\n";
    }

    return 0;
}
